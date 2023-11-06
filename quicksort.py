def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)

if __name__ == "__main__":
    user_input = input("Enter a list of numbers separated by spaces: ")
    input_list = [int(x) for x in user_input.split()]
    
    sorted_list = quicksort(input_list)
    
    print("Sorted list: ", sorted_list)
