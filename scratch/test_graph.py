from workflow import build_guardian_graph

def test():
    app = build_guardian_graph()
    print("Graph Nodes:", app.nodes.keys())
    # print("Graph Edges:", app.edges) # edges might not be easily accessible like this
    
    # Just print the config
    print("Graph setup completed.")

if __name__ == "__main__":
    test()
