#include <bits/stdc++.h>
using namespace std;

class error1 : public exception
{
private:
    string msg = "Error had Occured during the exceution of the program.\n";

public:
    error1(const string &msg_inp = "")
    {
        this->msg += msg_inp;
    }
    const char *what()
    {
        return (this->msg).c_str();
    }
};

class Graph
{
private:
    char type_storage = '\0';
    char type_graph = '\0';
    char type_weights = '\0';
    int nodes = 0; // Storage Number of the Nodes or the Vertices in the Graph
    int edges = 0; // Stores the number of the Edges in the Graph
    vector<vector<int>> adj;
    vector<vector<int>> lst;

public:
    /**
     *  @brief Creats the Constructor of the Class Graph
     *  @ingroup constructor
     *  @param  graph_type Defines whether the graph is directed type('D') or UndirectedType('U)
     *  @param  store_type  Defines the Types of Storage used for the Graph:  A=Adjacency matrix, L=List
     *  @param  weights  Defines the whether weights to edge has to taken as Input or not: I=Takes Input, N = Not Input(default=1)
     *  @return   returns the constructor
     */
    Graph(char graph_type = 'U', char store_type = 'A', char weights = 'N') : type_graph(graph_type),type_storage(store_type), type_weights(weights)
    {
        type_storage = store_type;
        type_graph = graph_type;
        if (graph_type == 'U')
        {
            if (store_type == 'A')
            {
                int n, m;
                cin >> n >> m;
                nodes = n;
                edges = m;
                if (weights == 'N')
                {
                    vector<vector<int>> adj_mat(n + 1);
                    for (int i = 0; i < n + 1; i++)
                    {
                        vector<int> vec(n + 1, -1);
                        adj_mat[i] = vec;
                    }
                    for (int i = 0; i < m; i++)
                    {
                        int e1, e2;
                        cin >> e1 >> e2;
                        adj_mat[e1][e2] = 1;
                        adj_mat[e2][e1] = 1;
                    }
                    adj = adj_mat;
                }
                else if (weights == 'I')
                {
                    vector<vector<int>> adj_mat(n + 1);
                    for (int i = 0; i < n + 1; i++)
                    {
                        vector<int> vec(n + 1, -1);
                        adj_mat[i] = vec;
                    }
                    for (int i = 0; i < m; i++)
                    {
                        int e1, e2, wt;
                        cin >> e1 >> e2 >> wt;
                        adj_mat[e1][e2] = wt;
                        adj_mat[e2][e1] = wt;
                    }
                    adj = adj_mat;
                }
                else
                    throw error1("Weights type must I(=Takes Input) or N(= Not Input(default=1)).");
            }
            else if (store_type == 'L')
            {
                int n, m;
                cin >> n >> m;
                nodes = n;
                edges = m;
                if (weights == 'N')
                {
                    vector<vector<int>> adj_mat(2 * m);
                    for (int i = 0; i < m; i++)
                    {
                        int e1, e2;
                        cin >> e1 >> e2;
                        vector<int> vec1(3, 1);
                        vec1[0] = e1;
                        vec1[1] = e2;
                        adj_mat[2 * i] = vec1;
                        vector<int> vec2(3, 1);
                        vec2[0] = e2;
                        vec2[1] = e1;
                        adj_mat[2 * i + 1] = vec2;
                    }
                    lst = adj_mat;
                }
                else if (weights == 'I')
                {
                    vector<vector<int>> adj_mat(2 * m);
                    for (int i = 0; i < m; i++)
                    {
                        int e1, e2, wt;
                        cin >> e1 >> e2 >> wt;
                        vector<int> vec1(3, wt);
                        vec1[0] = e1;
                        vec1[1] = e2;
                        adj_mat[2 * i] = vec1;
                        vector<int> vec2(3, wt);
                        vec2[0] = e2;
                        vec2[1] = e1;
                        adj_mat[2 * i + 1] = vec2;
                    }
                    lst = adj_mat;
                }
                else
                    throw error1("Weights type must I(=Takes Input) or N(= Not Input(default=1)).");
            }
            else
                throw error1("Storage Type must be A(=Adjacency matrix) or L(=List).");
        }
        else if (graph_type == 'D')
        {
            if (store_type == 'A')
            {
                int n, m;
                cin >> n >> m;
                nodes = n;
                edges = m;
                if (weights == 'N')
                {
                    vector<vector<int>> adj_mat(n + 1);
                    for (int i = 0; i < n + 1; i++)
                    {
                        vector<int> vec(n + 1, -1);
                        adj_mat[i] = vec;
                    }
                    for (int i = 0; i < m; i++)
                    {
                        int e1, e2;
                        cin >> e1 >> e2;
                        adj_mat[e1][e2] = 1;
                    }
                    adj = adj_mat;
                }
                else if (weights == 'I')
                {
                    vector<vector<int>> adj_mat(n + 1);
                    for (int i = 0; i < n + 1; i++)
                    {
                        vector<int> vec(n + 1, -1);
                        adj_mat[i] = vec;
                    }
                    for (int i = 0; i < m; i++)
                    {
                        int e1, e2, wt;
                        cin >> e1 >> e2 >> wt;
                        adj_mat[e1][e2] = wt;
                    }
                    adj = adj_mat;
                }
                else
                    throw error1("Weights type must I(=Takes Input) or N(= Not Input(default=1)).");
            }
            else if (store_type == 'L')
            {
                int n, m;
                cin >> n >> m;
                nodes = n;
                edges = m;
                if (weights == 'N')
                {
                    vector<vector<int>> adj_mat(m);
                    for (int i = 0; i < m; i++)
                    {
                        int e1, e2;
                        cin >> e1 >> e2;
                        vector<int> vec1(3, 1);
                        vec1[0] = e1;
                        vec1[1] = e2;
                        adj_mat[i] = vec1;
                    }
                    lst = adj_mat;
                }
                else if (weights == 'I')
                {
                    vector<vector<int>> adj_mat(m);
                    for (int i = 0; i < m; i++)
                    {
                        int e1, e2, wt;
                        cin >> e1 >> e2 >> wt;
                        vector<int> vec1(3, wt);
                        vec1[0] = e1;
                        vec1[1] = e2;
                        adj_mat[i] = vec1;
                    }
                    lst = adj_mat;
                }
                else
                    throw error1("Weight type must be I(=Takes Input) or N(=Not Input(default=1)).");
            }
            else
                throw error1("Storage Type must be A(=Adjacency matrix) or L(=List).");
        }
        else
            throw error1("Graph Type must be U(=Undirected Graph) or D(=Directed Graph).");
    }
    void print_graph()
    {
        if (type_graph == 'U')
        {
            if (type_storage == 'A')
            {
                for (int i = 1; i < nodes + 1; i++)
                {
                    for (int j = 1; j < nodes + 1; j++)
                    {
                        cout << adj[i][j] << " ";
                    }
                    cout << endl;
                }
            }
            else if (type_storage == 'L')
            {
                for (int i = 0; i < 2 * edges; i++)
                {
                    cout << lst[i][0] << " " << lst[i][1] << endl;
                }
            }
        }
        else if (type_graph == 'D')
        {
            if (type_storage == 'A')
            {
                for (int i = 1; i < nodes + 1; i++)
                {
                    for (int j = 1; j < nodes + 1; j++)
                    {
                        cout << adj[i][j] << " ";
                    }
                    cout << endl;
                }
            }
            else if (type_storage == 'L')
            {
                for (int i = 0; i < edges; i++)
                {
                    cout << lst[i][0] << " " << lst[i][1] << endl;
                }
            }
        }
    }

    vector<vector<int>> get_Adj_mat_from_lst(vector<vector<int>> lst,int Nodes){
        vector<vector<int>> adj_mat(Nodes+1);
        for(int i=0;i<Nodes+1;i++)
            adj_mat[i] = vector<int>(Nodes+1);
        for(int i=0;i<lst.size();i++){
            adj_mat[lst[i][0]][lst[i][1]] = lst[i][3];
        }
        return adj_mat;
    }

    vector<vector<int>> get_Adj_List_from_Adj_Matrix(vector<vector<int>> adj_mat){
        vector<vector<int>> al(adj_mat.size());
        for(int i=0;i<adj_mat.size();i++){
            int counter=0;
            for(int j=0;j<adj_mat.size();j++){
                if(!adj_mat[i][j]==-1)
                    counter++;
            }
            vector<int> vec(counter,0);
            for(int j=adj_mat.size()-1;j>=0;j--){
                if(!adj_mat[i][j]==-1){
                    vec[counter-1]=adj_mat[i][j];
                    counter--;
                }
            }
            al[i]=vec;
        }
        return al;
    }

    vector<vector<int>> get_Adj_List_from_List(vector<vector<int>> lst,int Nodes){
        return get_Adj_List_from_Adj_Matrix(get_Adj_mat_from_lst(lst,Nodes));
    }

    vector<vector<int>> get_Adj_List(){
        if(type_storage=='A') //if storage type is Adjacency matrix
            return get_Adj_List_from_Adj_Matrix(this->adj);
        else
            return get_Adj_List_from_List(this->lst,nodes);
    }

    vector<int> BFS(vector<vector<int>> adj_list,int start){
        vector<int> visited(adj_list.size(),0);
        queue<int> que;
        que.push(start);
        vector<int> out;
        for(int i=start;visited[i]!=1;i++){
            if(i==0){
                visited[0]=1;
                continue;
            }
            else if(i==adj_list.size()){
                i=0;
                continue;
            }
            else{
                out.push_back(que.front());
                for(int j=0;j<adj_list[que.front()].size();j++){
                    if(visited[adj_list[que.front()][j]]!=0){
                        que.push(adj_list[que.front()][j]);
                    }
                }
                que.pop();
            }
        }
    }

    vector<int> BFS(int start){
        return this->BFS(get_Adj_List(),start);
    }
};

int main()
{
    Graph g('U','A','N');
    g.print_graph();
    return 0;
}