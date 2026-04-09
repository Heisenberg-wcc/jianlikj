from train_q_mc import MCQTrainer

t = MCQTrainer()
for i in range(3):
    samples, info = t.play_episode(render=True)
    print(f"--- Game {i+1}: steps={info['steps']}, team1_win={info['team1_win']}, scores={info['team_scores']}")
    print()
