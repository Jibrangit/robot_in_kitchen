// BFS

// way 1 maintain vertex list and list of connection : vertex list = [1,2,3,4,5,6] connection list [ (1,3) (2,5) (4,5) ...]
// memory - O(v + c)
// connection search - o(c)

// way 2 maintain adjacency matrix
// memory - O(v*v)
// connection search - O(1)

// maintain adjacency list | [1 -> [2,3,4], [2->[5, 6, 7], ....]
// memory - O(v) - (look up)
// connection search - O(v)

// implement graph using Adjacency list
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <utility>
#include <set>

#include <algorithm>

class AdjacencyList
{
public:
    std::set<int> get_connections  (int pt)    {
        return map.find(pt).second;
    }

private:
    bool is_connected(int pt1, int pt2)
    {
        auto it = map.find(pt1) for (auto pt : *it)
        {
            if (pt == pt2)
                return true
        }
        return false;
    }
    std::unordered_map<int, std::unordered_set<int>> map;
};

std::vector<int> bfs(const int &st_pt, const int &ed_pt, const AdjacencyList &graph)
{
    std::vector<int> explored;
    std::queue<int> q;
    std::unordered_map<int, int> parents; // Child, Parent 
    parents[st_pt] = st_pt; 

    // Path every vertex should get a parent
    //
    // exploration part
    q.push(st_pt);
    while (!q.empty())
    {
        int cur_pt = q.front();
        if (cur_pt == ed_pt)
        {
        }
        q.pop();

        auto neighbors = graph.get_connections(cur_pt);
        for (auto neighbor : neighbors)
        {
            if (explored.find(neighbor) == explored.end())
            {
                q.push(neighbor);
                parents[neighbor] = cur_pt;
            }
        }
    }

    // path finding

    std::reverse(explored.begin(), explored.end());
    std::vector<int> path;
    path.push_back(explored[0]);

    for (auto pt : explored)
    {
        if (path.end())
    }
    isconnected(, pt)
        path.push_back(pt);
}