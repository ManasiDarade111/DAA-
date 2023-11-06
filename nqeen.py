def is_safe(board, row, col):
    # Check if there is a Queen in the same column
    for i in range(row):
        if board[i][col] == "Q":
            return False

    # Check upper-left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == "Q":
            return False

    # Check upper-right diagonal
    for i, j in zip(range(row, -1, -1), range(col, len(board))):
        if board[i][j] == "Q":
            return False

    return True

def solve_n_queens(n):
    def backtrack(row):
        if row == n:
            solutions.append(["".join(row) for row in board])
            return

        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = "Q"
                backtrack(row + 1)
                board[row][col] = "."

    solutions = []
    board = [["." for _ in range(n)] for _ in range(n)]
    backtrack(0)
    return solutions

def print_solutions(solutions):
    for solution in solutions:
        for row in solution:
            print(row)
        print()

if __name__ == "__main__":
    n = int(input("Enter the size of the chessboard: "))
    solutions = solve_n_queens(n)

    if solutions:
        print("Solutions to the N-Queens puzzle:")
        print_solutions(solutions)
    else:
        print("No solutions found.")
