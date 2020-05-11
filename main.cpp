#include <iostream>

#include "include/element.hpp"
#include "include/mesh.hpp"
#include "include/types.hpp"

#include <vtkUnstructuredGrid.h>
#include <vtkCell.h>
#include <vtkIntArray.h>
#include <vtkCellData.h>

#include <boost/property_tree/ptree.hpp>

#include "hddApi.hpp"
//#include "hddPythonApi.hpp"
#include <mpi.h>

using namespace Eigen;

int main(int argc, char **argv) 
{

  MPI_Init(&argc, &argv);
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD,&comm);


  std::string path2FemJsonFile = "fem.json";
  boost::property_tree::ptree root;
  boost::property_tree::read_json(path2FemJsonFile, root);

  int rank;

  MPI_Comm_rank(comm, &rank);
// ** H-D-D **
  HddApi hdd(comm);

  std::string path2OptJsonFile("");
  if (argc > 1)
    path2OptJsonFile = argv[1];
  else
    path2OptJsonFile = "../hddConf.json";

  std::cout << "path2OptJsonFile: " << path2OptJsonFile << '\n';

  hdd.ParseJsonFile(path2OptJsonFile);

//////////////////////
// Built-in mesher and
// assembler of A*x=b
  Mesh mesh;
  auto meshOpts = root.get_child("mesh");
  mesh.GenerateMesh(rank,meshOpts);

// decomposition
  mesh.DomainDecomposition(); // metis
  mesh.ExtractSubdomainMesh();

  // symbolic part
  {
    vtkUnstructuredGrid* subMesh = mesh.getSubdomainMesh();
    auto glbNum = subMesh->GetCellData()->GetArray(GLOBAL_NUMBERING);
    int nElemeSubdomain = subMesh->GetNumberOfCells();
    int nP = subMesh->GetCell(0)->GetNumberOfPoints();

    std::vector<int> glbIds;
    glbIds.resize(2 * nP);

    for (int iE = 0; iE < nElemeSubdomain; iE++)
    {
      for (int iP = 0; iP < nP; iP++)
      {
        int iGI = (int) glbNum->GetComponent(iE,iP);
        glbIds[2 * iP    ] = 2 * iGI;
        glbIds[2 * iP + 1] = 2 * iGI + 1;
      }
// ** H-D-D **
//#  loop over subdomain elements
//#  passing global DOFs element by element - metis decomposition
      hdd.SymbolicAssembling(glbIds.data(),glbIds.size());
    }
  }

// ** H-D-D ** global DOFs where Dirichlet BC is applied
  hdd.SetDirichletDOFs(mesh.getDirDOFs().data(),mesh.getDirDOFs().size());

// ** H-D-D ** creation mapping vectors etc ...
  int neqSubdomain = hdd.FinalizeSymbolicAssembling();


  {
    vtkUnstructuredGrid* subMesh = mesh.getSubdomainMesh();
    auto glbNum = subMesh->GetCellData()->GetArray(GLOBAL_NUMBERING);
    int nElemeSubdomain = subMesh->GetNumberOfCells();
    int nP = subMesh->GetCell(0)->GetNumberOfPoints();

    std::cout << "np: " << nP << '\n';

    std::vector<int> glbIds;
    glbIds.resize(2 * nP);

    MatrixXd K_loc;
    VectorXd f_loc;


    auto matOpts = root.get_child("material");

    double mat_E0     = matOpts.get<double>("YoungsModulus",1);
    double mat_mu     = matOpts.get<double>("poissonRatio",0.3);
    double mat_rho    = matOpts.get<double>("density",1);
    double mat_ratio  = matOpts.get<double>("ratio",1);



    auto MatId = subMesh->GetCellData()->GetArray(MATERIAL_ID);
    for (int iE = 0; iE < nElemeSubdomain; iE++)
    {
      auto cell = subMesh->GetCell(iE);
      Element *element;

      switch (cell->GetCellType()){
        case VTK_QUADRATIC_QUAD:
          element = new QUADRATIC_QUAD;
          break;
        case VTK_QUAD:
          element = new QUAD;
          break;
        default:
          continue;
      }

      int matLabel = MatId->GetTuple1(iE);

      double weight = (mat_ratio - 1.0) * matLabel + 1.0;

      double YoungModulus = mat_E0 * weight;
      double rho = mat_rho * weight;
      double mat_E_mu_rho[3] = {YoungModulus, mat_mu, rho};

      element->assembly_elasticity(K_loc, f_loc, cell, mat_E_mu_rho);

#if DBG > 4
      auto sv = linalg::svd0(K_loc);
      for (int i = 0; i < sv.size(); i++)
        std::cout << sv(i) << ' ';
      std::cout << '\n';
#endif


      for (int iP = 0; iP < nP; iP++)
      {
        int iGI = (int) glbNum->GetComponent(iE,iP);
        glbIds[iP     ] = 2 * iGI;
        glbIds[iP + nP] = 2 * iGI + 1;
      }
      std::vector<double> stdK(K_loc.data(),
          K_loc.data() + K_loc.size());
      std::vector<double> stdF(f_loc.data(),
          f_loc.data() + f_loc.size());

// ** H-D-D ** loop over elements - passing local stiffness matrices
      hdd.NumericAssembling(glbIds.data(),stdK.data(),stdF.data(),glbIds.size());
    }
  }

// ** H-D-D ** numerical factorization etc.
  hdd.FinalizeNumericAssembling();

// ** H-D-D ** solving 
  Eigen::MatrixXd solution = Eigen::MatrixXd::Zero(neqSubdomain,1);
  hdd.Solve(solution.data(),solution.rows(), solution.cols());

  auto otpOpts = root.get_child("outputs");
  bool saveBuiltInMesh = otpOpts.get<bool>("saveEachSubdomainMesh",false);
  std::cout << "saveBuiltInMes: " << saveBuiltInMesh << '\n';
  if (saveBuiltInMesh)
  {
    mesh.addSolution(mesh.getSubdomainMesh(),solution);
    mesh.SaveDecomposedMesh();
  }


// ** H-D-D ** finalizing 
  hdd.Finalize();

  MPI_Comm_free(&comm);
  MPI_Finalize();

  return 0;
}

