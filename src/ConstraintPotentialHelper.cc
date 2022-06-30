
// Fix for Boost.

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
// Test for GCC >= 4.3.0 
#if GCC_VERSION > 40300
#define BOOST_NO_HASH
#endif

#include <thrust/device_vector.h> // HACK to recognize cuda data_types!

#include "rumd/ConstraintPotentialHelper.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>

using namespace boost;

// Each edge is associated with a weight = the bond length.
typedef property<edge_weight_t, float> EdgeProperty;
// Create an undirected graph with edge weights. No vertex properties.
typedef adjacency_list<vecS, vecS, undirectedS, no_property, EdgeProperty> Graph;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_descriptor Source;

// A visitor class implementing actions to be taken during DFS.
class ConstraintVisitor: public default_dfs_visitor{

private:
  ConstraintVisitor& operator=(const ConstraintVisitor&);
  
  int counter; // Registers which component the constraints belong to.
  std::list<float4> &constraintList;
  
public:
  ConstraintVisitor( std::list<float4> &c ) : counter(-1), constraintList(c){};
  
  void start_vertex( Source s, const Graph& g ){ 
    if(degree(s,g) != 0)
      counter++; 
    
    // Removes compiler warnings:
    num_vertices(g); 
    // get(vertex_color, g, s);
  };
  
  void examine_edge( Edge e, const Graph& g ){
    float particleI = (float) source(e,g); 
    float particleJ = (float) target(e,g);
    float bondLength = get(edge_weight, g, e);
    float4 constraintInfo = { particleI, particleJ, bondLength, static_cast<float>(counter) };

    if( particleI > particleJ ){
      constraintInfo.x = particleJ; constraintInfo.y = particleI;
    }

    constraintList.push_back(constraintInfo);
  };
};

// Removes duplicates from list due to DFS.
void RemoveDuplicatesFromList( std::list<float4> &c ){
  for(std::list<float4>::iterator it = c.begin(); it != c.end(); it++){
    std::list<float4>::iterator its = it;
    
    for( ++its; its != c.end(); ){
      if( ((unsigned) (*it).x) == ((unsigned) (*its).x) && ((unsigned) (*it).y) == ((unsigned) (*its).y) )
	its = c.erase(its);
      else
	its++;
    } 
  }
}

void BuildConstraintGraph( MoleculeData* M, std::list<float4> &constraintList ){
  Graph G; ConstraintVisitor vis(constraintList);
  
  // Read constraints and build a constraint graph.
  for( unsigned i = 0; i < M->GetNumberOfBonds(); i++){
    unsigned userBondType = M->h_btlist[i].x;
    unsigned internalBondType = M->h_btlist_int[userBondType].x;
      
    if( internalBondType == 2 ){
      unsigned particleI = M->h_blist[i].x;
      unsigned particleJ = M->h_blist[i].y;
      float bondLength = M->h_bplist[userBondType].x;
	
      add_edge(particleI, particleJ, EdgeProperty(bondLength), G);
    }
  }

  // Parse the constraint graph.
  depth_first_search(G, visitor(vis));
  RemoveDuplicatesFromList(constraintList);
}
