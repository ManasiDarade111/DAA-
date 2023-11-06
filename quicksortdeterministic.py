def deterministic_quicksort(arr):
    if len(arr) <= 1:
        return arr  # Base case: if the array has one element or is empty, return it as it's already sorted.

    pivot = choose_pivot(arr)  # Choose a pivot using a deterministic method.
    less, equal, greater = partition(arr, pivot)  # Partition the array into three parts: less than, equal to, and greater than the pivot.

    sorted_less = deterministic_quicksort(less)  # Recursively sort the "less than" part.
    sorted_greater = deterministic_quicksort(greater)  # Recursively sort the "greater than" part.

    return sorted_less + equal + sorted_greater  # Combine the three parts to get the final sorted array.

def choose_pivot(arr):
    # Choose the pivot as the median of the first, middle, and last elements.
    first, middle, last = arr[0], arr[len(arr) // 2], arr[-1]
    median = sorted([first, middle, last])[1]  # Find the median value.
    return median

def partition(arr, pivot):
    less, equal, greater = [], [], []

    for element in arr:
        if element < pivot:
            less.append(element)
        elif element == pivot:
            equal.append(element)
        else:
            greater.append(element)

    return less, equal, greater

if __name__ == "__main__":
    user_input = input("Enter a list of numbers separated by spaces: ")
    input_list = [int(x) for x in user_input.split()]

    sorted_list = deterministic_quicksort(input_list)

    print("Sorted list: ", sorted_list)
