import csv, sys
path = sys.argv[1]
for d in csv.DictReader(open(path)):
    print(f"seed={d['seed']}  leader_final_x={d['leader_final_x']}  "
          f"fallback={d['fallback_events']}  safety={d['safety_interventions']}  "
          f"collision={d['collision_count']}  boundary={d['boundary_violation_count']}  "
          f"reached={d['reached_goal']}  team_reached={d['team_goal_reached']}")
