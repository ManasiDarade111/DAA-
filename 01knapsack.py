def knapsack(items, total_capacity):
    n = len(items)
    dp = [[0 for _ in range(total_capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(total_capacity + 1):
            weight, value, name = items[i - 1]
            if weight <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)
            else:
                dp[i][w] = dp[i - 1][w]

    selected_objects = []
    i, w = n, total_capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            weight, _, name = items[i - 1]
            selected_objects.append(name)
            w -= weight
        i -= 1

    return dp[n][total_capacity], selected_objects

if __name__ == "__main__":
    n = int(input("Enter the number of objects: "))
    items = []
    for i in range(n):
        name = input(f"Enter the name of object {i + 1}: ")
        value = int(input(f"Enter the value of object {i + 1}: "))
        weight = int(input(f"Enter the weight of object {i + 1}: "))
        items.append((weight, value, name))

    total_capacity = int(input("Enter the total knapsack capacity: "))

    max_value, selected_objects = knapsack(items, total_capacity)

    print("Maximum value for", total_capacity, "weight is", max_value)
    print("Objects added to the knapsack:")
    for obj in selected_objects:
        print(obj)