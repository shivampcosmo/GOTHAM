import sys
# try:
subsel_type = sys.argv[2]
add_space_token = bool(ast.literal_eval(sys.argv[1]))
grid_sbox = int(ast.literal_eval(sys.argv[0]))
# except:
#     subsel_type = 'all'
#     add_space_token = False
#     grid_sbox = 32

# try:
learning_rate = float(ast.literal_eval(sys.argv[3]))
max_iters = int(ast.literal_eval(sys.argv[4]))
# except:
#     learning_rate = 3e-4
#     max_iters = 1500

print(subsel_type, add_space_token, grid_sbox, learning_rate, max_iters)
# try:
