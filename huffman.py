import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_map):
    # Create the list of the heap which store the value
    heap = [HuffmanNode(char, freq) for char, freq in freq_map.items()]

    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        internal_node = HuffmanNode(None, left.freq + right.freq)
        internal_node.left = left
        internal_node.right = right

        heapq.heappush(heap, internal_node)

    return heap[0]

# generate the huffman code for the particular Char
def huffman_codes(root, current_code, result):
    if root is None:
        return

    if root.char is not None:
        result[root.char] = current_code

    huffman_codes(root.left, current_code + '0', result)
    huffman_codes(root.right, current_code + '1', result)

def generate_huffman_codes(characters, frequencies):
    if len(characters) != len(frequencies):
        return None

    freq_map = {char: freq for char, freq in zip(characters, frequencies)}

    root = build_huffman_tree(freq_map)
    huffman_map = {}
    huffman_codes(root, '', huffman_map)

    return huffman_map

if __name__ == "__main__":
    characters = ['a', 'b', 'c', 'd', 'e','f','g']
    frequencies = [45, 13, 12, 16, 9,50,0]

    huffman_codes = generate_huffman_codes(characters, frequencies)

    if huffman_codes:
        print("Character Huffman Codes:")
        for char, code in huffman_codes.items():
            print(f"{char}: {code}")
    else:
        print("Invalid input: characters and frequencies lists must have the same length.")
