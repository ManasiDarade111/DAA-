def calculate_fibonacci(index):
    if index <= 0:
        return 0
    elif index == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, index + 1):
            a, b = b, a + b
        return b

def fibonacci_series(index):
    if index <= 0:
        return [0]
    elif index == 1:
        return [0, 1]
    else:
        series = [0, 1]
        while series[-1] + series[-2] <= calculate_fibonacci(index):
            series.append(series[-1] + series[-2])
        return series

def main():
    try:
        index = int(input("Enter the index to calculate Fibonacci: "))
        if index < 0:
            print("Index should be a non-negative integer.")
            return

        fib_number = calculate_fibonacci(index)
        fib_series = fibonacci_series(index)

        print(f"Fibonacci number at index {index}: {fib_number}")
        print(f"Fibonacci series up to index {index}: {fib_series}")
        print(f"Step count: {len(fib_series) - 2}")  # Subtract 2 to exclude 0 and 1
    except ValueError:
        print("Invalid input. Please enter a valid non-negative integer.")

if __name__ == "__main__":
    main()