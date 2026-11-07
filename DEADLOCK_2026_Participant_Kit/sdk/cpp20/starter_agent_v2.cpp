// Deterministic public-observation baseline. Search covers at most 400 cells.
// Hidden orders and collisions are unknown; this policy cannot ensure survival.
#include "agent_sdk.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <queue>
#include <set>
#include <tuple>

namespace {
using namespace deadlock;
using Point = std::pair<int, int>;
struct Step {
  int dx, dy;
  Direction direction;
};
constexpr std::array<Step, 4> steps = {{{0, -1, Direction::North},
                                        {1, 0, Direction::East},
                                        {0, 1, Direction::South},
                                        {-1, 0, Direction::West}}};
std::map<int, int> last_defend;
std::pair<std::string, int> match;
int last_tick = 0;
template <class T> Point point(const T &u) { return {u.x, u.y}; }
int distance(Point a, Point b) {
  return std::abs(a.first - b.first) + std::abs(a.second - b.second);
}
int center_distance(const Observation &s, Point p) {
  int best = 40;
  if (!s.phase_four)
    return 0;
  for (const auto &c : s.phase_four->safe_zone_center_tiles)
    best = std::min(best, distance(p, point(c)));
  return best;
}
int radius(const Observation &s) {
  if (!s.phase_four)
    return 40;
  const auto &phase = *s.phase_four;
  if (phase.next_contraction &&
      s.tick >= phase.next_contraction->warning_at_tick)
    return std::min<int>(phase.safe_radius,
                         phase.next_contraction->safe_radius);
  return phase.safe_radius;
}
struct Hazards {
  std::set<Point> seals, danger, pulse;
};
Hazards hazards(const Observation &s) {
  Hazards h;
  for (const auto &w : s.warden_telegraphs)
    for (const auto &p : w.tiles) {
      // Observation tick is the resolution being decided, not the last tick.
      if (w.kind == TelegraphKind::Seal && w.starts_at_tick <= s.tick &&
          s.tick <= w.ends_after_tick)
        h.seals.insert(point(p));
      if ((w.kind == TelegraphKind::Pulse || w.kind == TelegraphKind::Hunt) &&
          s.tick <= w.ends_after_tick && w.starts_at_tick <= int(s.tick) + 1)
        h.danger.insert(point(p));
      if (w.kind == TelegraphKind::Pulse && w.starts_at_tick <= s.tick &&
          s.tick <= w.ends_after_tick)
        h.pulse.insert(point(p));
    }
  return h;
}
std::set<Point> blocked_tiles(const Observation &s, const UnitObservation &u,
                              const Hazards &h,
                              const std::set<Point> &reserved) {
  auto b = h.seals;
  b.insert(reserved.begin(), reserved.end());
  for (const auto &t : s.terrain)
    if (t.kind == TerrainKind::Wall || t.kind == TerrainKind::ControlRelay)
      b.insert(point(t));
  for (const auto &r : s.relays)
    b.insert(point(r));
  for (const auto &v : s.team.units)
    if (v.hp > 0 && v.id != u.id)
      b.insert(point(v));
  for (const auto &v : s.visible_enemies)
    if (v.hp > 0 && v.id != u.id)
      b.insert(point(v));
  return b;
}
struct Move {
  Direction direction;
  int count;
  std::vector<Point> traversed;
};
using Goal = std::pair<int, Point>;
std::optional<Move> route(const Observation &s, const UnitObservation &u,
                          const std::vector<Goal> &goals,
                          const std::set<Point> &blocked,
                          const std::set<Point> &danger) {
  // Legal straight MOVE edges include both Strider substeps. PULSE damages
  // endpoints after movement; it does not block intermediate cells.
  using Node = std::tuple<int, int, int>;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> queue;
  const auto origin = point(u);
  std::map<Point, int> costs{{origin, 0}};
  std::map<Point, std::optional<Move>> first{{origin, std::nullopt}};
  queue.emplace(0, u.y, u.x);
  while (!queue.empty()) {
    const auto [cost, y, x] = queue.top();
    queue.pop();
    if (cost != costs.at({x, y}))
      continue;
    for (const auto &step : steps) {
      std::vector<Point> traversed;
      for (int n = 1; n <= (u.kind == UnitKind::Strider ? 2 : 1); ++n) {
        Point p{x + step.dx * n, y + step.dy * n};
        if (p.first < 0 || p.first >= 20 || p.second < 0 || p.second >= 20 ||
            blocked.contains(p))
          break;
        traversed.push_back(p);
        int extra = 10 + (danger.contains(p) ? 40 : 0);
        if (s.phase_four && center_distance(s, p) > s.phase_four->safe_radius)
          extra += 20;
        int candidate = cost + extra;
        if (!costs.contains(p) || candidate < costs.at(p)) {
          costs[p] = candidate;
          first[p] =
              Point{x, y} == origin
                  ? std::optional<Move>(Move{step.direction, n, traversed})
                  : first.at({x, y});
          queue.emplace(candidate, p.second, p.first);
        }
      }
    }
  }
 
  std::optional<std::tuple<int, int, int, int, int>> best;
  std::optional<Move> answer;

  for (const auto &[rank, p] : goals) {
    if (costs.contains(p) && !blocked.contains(p) &&
        !danger.contains(p)) {

      int strategic_score = rank * 30 + costs.at(p);

      auto key = std::make_tuple(
          strategic_score,
          center_distance(s, p),
          costs.at(p),
          p.second,
          p.first
      );

      if (!best || key < *best) {
        best = key;
        answer = first.at(p);
      }
    }
  }

  return answer;
                          }
std::vector<UnitObservation> adjacent(const Observation &s,
                                      const UnitObservation &u) {
  std::vector<UnitObservation> enemies;
  for (const auto &e : s.visible_enemies)
    if (e.hp > 0 && !e.immune && distance(point(u), point(e)) == 1)
      enemies.push_back(e);
  std::sort(enemies.begin(), enemies.end(), [](const auto &a, const auto &b) {
    return std::tie(a.hp, a.id) < std::tie(b.hp, b.id);
  });
  return enemies;
}
int incoming(const Observation &s, const UnitObservation &u) {
  if (u.immune)
    return 0;
  int damage = 0;
  std::map<int, int> team_hits;
  for (const auto &e : adjacent(s, u)) {
    damage += e.kind == UnitKind::Bulwark ? 2 : 1;
    if (e.team_id)
      ++team_hits[*e.team_id];
  }
  for (const auto &[id, count] : team_hits)
    if (count >= 2)
      ++damage;
  return damage;
}
bool capture_ok(const Observation &s, const UnitObservation &u,
                const RelayObservation &r) {
  // A neutral relay reports contested even for OUR lone adjacent unit.
  // Interpret this public flag relative to ownership and visible rivals.
  bool rival = std::any_of(
      s.visible_enemies.begin(), s.visible_enemies.end(), [&](const auto &e) {
        return e.hp > 0 && e.team_id && e.team_id != s.team.id &&
               distance(point(e), point(r)) <= 1;
      });
  const bool friendly_adjacent = std::any_of(
      s.team.units.begin(), s.team.units.end(), [&](const auto &ally) {
        return ally.hp > 0 && distance(point(ally), point(r)) <= 1;
      });
  // A remote public contest with no adjacent ally is another team's presence.
  // Avoid that acquisition route for now, and reconsider on the next tick.
  const bool remote_contest = r.contested && !friendly_adjacent;
  return r.owner_team_id != s.team.id && !u.immune && !rival && !remote_contest;
}
std::vector<Goal> goals_for(const Observation &s, const UnitObservation &u,
                            int flux, const std::set<int> &claimed) {
  std::vector<Goal> goals;
 
for (const auto &r : s.relays) {
    if (claimed.contains(r.id) || !capture_ok(s, u, r))
        continue;

    bool assigned_to_ally = false;

    int my_distance = distance(point(u), point(r));

    for (const auto &ally : s.team.units) {
        if (ally.hp <= 0 || ally.id == u.id)
            continue;

        int ally_distance = distance(point(ally), point(r));

        if (ally_distance < my_distance ||
            (ally_distance == my_distance && ally.id < u.id)) {
            assigned_to_ally = true;
            break;
        }
    }

    if (!assigned_to_ally) {
        for (const auto &step : steps) {
            goals.push_back({
                r.owner_team_id ? 1 : 0,
                {int(r.x) + step.dx, int(r.y) + step.dy}
            });
        }
    }
}

  if (flux < 14)
    for (const auto &w : s.wells)
      if (w.charged)
        goals.push_back({flux >= 2 ? 2 : -1, point(w.position)});
  if (!u.immune)
    for (const auto &e : s.visible_enemies)
      if (e.hp > 0 && !e.immune)
        for (const auto &step : steps)
          goals.push_back({3, {int(e.x) + step.dx, int(e.y) + step.dy}});
  std::erase_if(goals, [&](const auto &g) {
    auto p = g.second;
    return p.first < 0 || p.first >= 20 || p.second < 0 || p.second >= 20 ||
           center_distance(s, p) > radius(s);
  });
  return goals;
}
struct Decision {
  UnitAction action;
  int cost;
  std::vector<Point> moved;
  std::optional<int> relay;
};
Decision moving(const UnitObservation &u, const Move &m) {
  return {MoveAction{u.id, m.direction, static_cast<std::uint8_t>(m.count)}, 0,
          m.traversed, std::nullopt};
}
Decision decide_unit(const Observation &s, const UnitObservation &u, int flux,
                     const Hazards &h, const std::set<Point> &reserved,
                     const std::set<int> &claimed, int harvested_flux) {
  const auto p = point(u);
  const auto blocked = blocked_tiles(s, u, h, reserved);
  const int threat = incoming(s, u);
  const bool can_defend =
      !u.immune && flux >= 1 &&
      (!last_defend.contains(u.id) || last_defend.at(u.id) != int(s.tick) - 1);
  const int hazard = u.immune
                         ? 0
                         : (h.pulse.contains(p) ? 2 : 0) +
                               (s.phase_four && center_distance(s, p) >
                                                    s.phase_four->safe_radius
                                    ? s.phase_four->hazard_damage
                                    : 0);
  const bool emergency =
      h.danger.contains(p) || center_distance(s, p) > radius(s);
  // Combat precedes movement; preserve a life that moving cannot save.
  if (can_defend && threat >= u.hp && threat - 2 < u.hp)
    return {DefendAction{u.id}, 1, {}, std::nullopt};
  if (emergency) {
    std::vector<Goal> goals;
    for (int y = 0; y < 20; ++y)
      for (int x = 0; x < 20; ++x)
        if (center_distance(s, {x, y}) <= radius(s))
          goals.push_back({0, {x, y}});
    const auto move = route(s, u, goals, blocked, h.danger);
    if (move && !h.danger.contains(move->traversed.back()))
      return moving(u, *move);
    if (can_defend && threat + hazard > 0)
      return {DefendAction{u.id}, 1, {}, std::nullopt};
  }
  auto enemies = adjacent(s, u);
  if (can_defend && threat + hazard > 0 &&
      (u.hp <= threat + hazard + 1 || threat >= 3 || flux < 2))
    return {DefendAction{u.id}, 1, {}, std::nullopt};
  
if (!u.immune && !enemies.empty() && flux >= 2) {
    auto target = enemies.front();
    int best_support = -1;

    for (const auto &enemy : enemies) {
        int support = 0;

        for (const auto &ally : s.team.units) {
            if (ally.hp > 0 && !ally.immune &&
                distance(point(ally), point(enemy)) == 1) {
                ++support;
            }
        }

        if (support > best_support ||
            (support == best_support &&
             std::make_tuple(enemy.hp, enemy.id) <
             std::make_tuple(target.hp, target.id))) {
            best_support = support;
            target = enemy;
        }
    }

    return {AttackAction{u.id, target.id}, 2, {}, std::nullopt};
}

  if (!emergency) {
    auto relays = s.relays;
    std::sort(relays.begin(), relays.end(),
              [](const auto &a, const auto &b) { return a.id < b.id; });
    const bool reserve = std::any_of(
        s.team.units.begin(), s.team.units.end(), [&](const auto &v) {
          return v.id != u.id && v.hp > 0 && incoming(s, v) >= v.hp;
        });
    for (const auto &r : relays)
      if (!claimed.contains(r.id) && capture_ok(s, u, r) &&
          distance(p, point(r)) == 1 && flux >= 1 &&
          !(flux == 1 && (u.hp <= 2 || reserve)))
        return {CaptureAction{u.id, r.id}, 1, {}, r.id};
    if (flux + harvested_flux < 14)
      for (const auto &w : s.wells)
        if (w.charged && point(w.position) == p)
          return {HarvestAction{u.id}, 0, {}, std::nullopt};
    if (auto move =
            route(s, u, goals_for(s, u, flux, claimed), blocked, h.danger))
      return moving(u, *move);
  }
  return {WaitAction{u.id}, 0, {}, std::nullopt};
}
} // namespace

deadlock::ActionBundle decide(const deadlock::Observation &state) {
  using namespace deadlock;
  const auto identity = std::make_pair(state.match_id, int(state.team.id));
  if (match != identity || state.tick < last_tick)
    last_defend.clear();
  match = identity;
  last_tick = state.tick;
  const auto h = hazards(state);
  auto units = state.team.units;
  std::erase_if(units, [](const auto &u) { return u.hp <= 0; });
  auto rank = [&](const auto &u) {
    return std::make_tuple(incoming(state, u) >= u.hp && !u.immune ? 0 : 1,
                           h.danger.contains(point(u)) ||
                                   center_distance(state, point(u)) >
                                       radius(state)
                               ? 0
                               : 1,
                           u.id);
  };
  std::sort(units.begin(), units.end(),
            [&](const auto &a, const auto &b) { return rank(a) < rank(b); });
  ActionBundle result;
  int flux = state.team.flux;
  int harvested_flux = 0;
  std::set<Point> reserved;
  std::set<int> claimed;
  for (std::size_t i = 0; i < std::min<std::size_t>(2, units.size()); ++i) {
    const auto &u = units[i];
    auto d = decide_unit(state, u, flux, h, reserved, claimed, harvested_flux);
    result.priority.push_back(u.id);
    flux -= d.cost;
    if (std::holds_alternative<HarvestAction>(d.action))
      harvested_flux += std::min(3, 14 - flux - harvested_flux);
    if (std::holds_alternative<DefendAction>(d.action))
      last_defend[u.id] = state.tick;
    if (d.relay)
      claimed.insert(*d.relay);
    reserved.insert(d.moved.begin(), d.moved.end());
    result.actions.push_back(std::move(d.action));
  }
  return result;
}
int main() { return deadlock::run(decide); }
