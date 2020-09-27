import java.io.*;
import java.util.*;
import java.util.concurrent.ForkJoinPool;

public class ParallelDijkstra {
    public static final String OUTPUT_FILENAME = "ParallelDijkstra.output";
    public static final String INFINITY_DISTANCE_NAME = "INF";

    private static BufferedReader br;
    private static StringTokenizer tok;

    private static String readToken() throws IOException {
        while (tok == null || !tok.hasMoreTokens()) {
            tok = new StringTokenizer(br.readLine());
        }
        return tok.nextToken();
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 2) throw new RuntimeException("Please specify filename and source node.");

        // 0. Initialize
        br = new BufferedReader(new FileReader(new File(args[0])));
        int sourceNode = Integer.valueOf(args[1]);
        int[] processesNumbers = {1, 2, 3, 4, 5, 6};
        if (args.length < 3) {
            System.out.println("Number of processes is not specified. calculating for 1-6 processes");
        } else {
            processesNumbers = new int[1];
            processesNumbers[0] = Integer.valueOf(args[2]);
        }
        int edgesNum = Integer.valueOf(readToken());
        List<NodeInfo> nodes = new ArrayList<>(edgesNum);
        for (int i = 0; i < edgesNum; i++) {
            // negative distance means there is no way to this node found
            nodes.add(new NodeInfo(i, (sourceNode == i) ? 0L : -1L, false));
        }
        for (int i = 0; i < edgesNum; i++) {
            for (int j = 0; j < edgesNum; j++) {
                String token = readToken();
                int distance = token.equals(INFINITY_DISTANCE_NAME) ? -1 : Integer.valueOf(token);
                if ((distance < 0 && i != j) || i > j) continue; // i > j to avoid duplication of edges
                NodeInfo node1 = nodes.get(i),
                        node2 = nodes.get(j);
                Edge edge = new Edge(node1, node2, distance);

                node1.edges.add(edge);
                node2.edges.add(edge);
            }
        }
        br.close();

        // 1. Calculating
        for (int processesNumber : processesNumbers) {
            long start = System.currentTimeMillis();
            ForkJoinPool forkJoinPool = new ForkJoinPool(processesNumber);
            while (true) {
                // 1.1 Get closest node or left if all visited
                NodeInfo currentNode = forkJoinPool.submit(() -> nodes.parallelStream()
                        .filter(node -> !node.isVisited && node.distance >= 0)
                        .min(Comparator.comparingLong(node -> node.distance))
                        .orElse(null)).join();

                if (currentNode == null) {
                    break;
                }

                // 1.2 Visit neighbors to give them shorter distance
                currentNode.isVisited = true;
                forkJoinPool.submit(() -> currentNode.edges.parallelStream()
                        .forEach(edge -> {
                            long newDistance = edge.distance + currentNode.distance;
                            NodeInfo neighbor = edge.getOtherNode(currentNode);
                            if (neighbor.distance < 0 || neighbor.distance > newDistance) {
                                neighbor.distance = newDistance;
                            }
                        })).join();
            }
            long end = System.currentTimeMillis();
            System.out.println(processesNumber + ";" + edgesNum + ";" + (end - start));
            // 1.3 clearing calculated values
            for (NodeInfo node : nodes) {
                node.distance = (node.number == sourceNode) ? 0 : -1;
                node.isVisited = false;
            }
        }

        // 2. Printing result
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(OUTPUT_FILENAME)));
        for (NodeInfo nodeInfo : nodes) {
            bw.write(((nodeInfo.distance < 0) ? INFINITY_DISTANCE_NAME : nodeInfo.distance) + "\n");
        }
        bw.close();
    }

    private static class NodeInfo {
        int number;
        long distance;
        boolean isVisited;
        List<Edge> edges;

        NodeInfo(int number, long distance, boolean isVisited) {
            this.number = number;
            this.distance = distance;
            this.isVisited = isVisited;
            edges = new ArrayList<>();
        }
    }

    private static class Edge {
        NodeInfo node1, node2;
        int distance;

        Edge(NodeInfo node1, NodeInfo node2, int distance) {
            this.node1 = node1;
            this.node2 = node2;
            this.distance = distance;
        }

        NodeInfo getOtherNode(NodeInfo source) {
            return (source == node1) ? node2 : node1;
        }
    }
}
