from itertools import combinations


hyperedges = {
    ('A', 'B', 'C', 'D', 'E'): 10,
    ('F', 'G', 'H', 'I', 'J'): 15,
    # ...
}


def max_damage(characters, num_chars):
    combos = [combo for r in range(1, 41) for combo in combinations(hyperedges.keys(), r)]

    # 各組み合わせに対してダメージを計算
    max_damage = 0
    best_combo = None
    for combo in combos:
        # ハイパーエッジが重ならないことを確認
        if len(set(sum(combo, ()))) == len(sum(combo, ())):
            damage = sum(hyperedges[edge] for edge in combo)
            if damage > max_damage:
                max_damage = damage
                best_combo = combo
                if max_damage > 500000:  # ダメージが500000を超えたら終了
                    break

    return best_combo, max_damage

print(calculate_max_damage(hyperedges))