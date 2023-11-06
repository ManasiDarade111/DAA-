def fractional_knapsack(objects, total_capacity):
    # Calculate the value-to-weight ratio for each object
    value_per_weight = [(obj[0], obj[1], obj[2], obj[2] / obj[1]) for obj in objects]
    
    # Sort the objects by the value-to-weight ratio in descending order
    value_per_weight.sort(key=lambda x: x[3], reverse=True)
    
    max_profit = 0  # Maximum profit obtained
    knapsack = []   # List to store the selected objects
    
    for obj in value_per_weight:
        name, weight, value, ratio = obj
        
        if total_capacity >= weight:
            max_profit += value
            total_capacity -= weight
            knapsack.append((name, weight))
        else:
            fraction = total_capacity / weight
            max_profit += fraction * value
            knapsack.append((name, fraction * weight))
            break
    
    return max_profit, knapsack

def main():
    n = int(input("Enter the number of objects: "))
    objects = []
    for i in range(n):
        name = input(f"Enter the name of object {i+1}: ")
        weight = float(input(f"Enter the weight of object {i+1}: "))
        value = float(input(f"Enter the value of object {i+1}: "))
        objects.append((name, weight, value))

    total_capacity = float(input("Enter the total knapsack capacity: "))
    
    max_profit, knapsack = fractional_knapsack(objects, total_capacity)
    
    print("Maximum profit:", max_profit)
    print("Objects added to the bag:")
    for name, weight in knapsack:
        print(f"{name} - {weight} units")

if __name__ == "__main__":
    main()
