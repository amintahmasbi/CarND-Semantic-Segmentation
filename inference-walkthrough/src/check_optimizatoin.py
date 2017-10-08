from graph_utils import load_graph

sess, base_ops = load_graph('my_graph.pb')
print(len(base_ops)) # 2026
sess, frozen_ops = load_graph('my_frozen_graph.pb')
print(len(frozen_ops)) # 276
sess, optimized_ops = load_graph('my_optimized_graph.pb')
print(len(optimized_ops)) # 231
sess, eightbit_ops = load_graph('my_eightbit_graph.pb')
print(len(eightbit_ops)) # 477
