#include <Rcpp.h>
#include <omp.h>
#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <string.h>

//GENENUM stands for rows of SNPs, INDIVIDUALS stands for number of individuals
#define GENENUM 5000
#define INDIVIDUALS 1000

using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]

void generateR_improve8(SEXP filename_)
{
  //Input
  string filename = as<string>(filename_);
  
  //Declare variables
  FILE *pFile;
	int i,j;
	char str[1000];
	float *SNP = new float[3*INDIVIDUALS];
	float *SNP_ = new float[INDIVIDUALS];
	float *SNPFrequency = new float[INDIVIDUALS];
  float *SNPFrequency_pl = new float[INDIVIDUALS];
  float *v_m = new float[GENENUM*INDIVIDUALS];
  float *v_r = new float[GENENUM*INDIVIDUALS];
  float *M = new float[INDIVIDUALS*INDIVIDUALS];
  float *R = new float[INDIVIDUALS*INDIVIDUALS];

  //open file
  pFile = fopen(filename.c_str(),"r");

  //Initialize v_m and v_r  
  memset(v_m,0,GENENUM*INDIVIDUALS*sizeof(float));
  memset(v_r,0,GENENUM*INDIVIDUALS*sizeof(float));
  
	//Initialize M and R  
  memset(M,0,INDIVIDUALS*INDIVIDUALS*sizeof(float));
  memset(R,0,INDIVIDUALS*INDIVIDUALS*sizeof(float));
  
  //Declare variables
	float pl = 0, pl_ = 0, VG = 0, IV = 0;
	int num_missing = 0;

  //Deal with one loci at a time
	for(i=0;i<GENENUM;i++)
	{
    //Skip first five strings
		fscanf(pFile,"%s %s %s %s %s",str,str,str,str,str);
		for(j=0;j<3*INDIVIDUALS;j++){
			fscanf(pFile, "%f", &SNP[j]);
		}
    
		//Calculate the pl of the genotype, and the percentage of individuals with missing genotype
		pl = 0;
		pl_ = 0;
		num_missing = 0;
		VG = 0;

		for(j=0;j<INDIVIDUALS;j++)
		{
      SNP_[j] = SNP[3*j]+SNP[3*j+1]+SNP[3*j+2];
    	SNPFrequency[j] = SNP[3*j+1]/SNP_[j] + 2*SNP[3*j+2]/SNP_[j];
			if(SNP_[j] > 0.1)
			{
				pl += SNPFrequency[j];
				pl_ += 2;
			}	
			else
			{
				num_missing++;
			}
		}
		pl /= pl_;
    
    //Calculate some part of computations for R matrix in advance
    for(j=0;j<INDIVIDUALS;j++)
    {
    	SNPFrequency_pl[j] = SNPFrequency[j] - 2 * pl;
    }
 
    //Calculate VG and VI
		for(j=0;j<INDIVIDUALS;j++)
		{
			if(SNP_[j] > 0.1)
			{
				VG += SNP_[j]*(SNP[3*j+1]/SNP_[j]+4*SNP[3*j+2]/SNP_[j]-SNPFrequency[j]*SNPFrequency[j])+(1-SNP_[j])*2*pl*(1-pl)+(1-SNP_[j])*SNP_[j]*SNPFrequency_pl[j]*SNPFrequency_pl[j];
			}
		}
		IV = 1 - VG/(2*pl*(1-pl)*(INDIVIDUALS-num_missing));

		//see if the percentage of individuals with missing genotype is above 75%
		if(1.0*num_missing/INDIVIDUALS > 0.75)
			continue;
		//see if minor allele frequency is less than 0.05
		if((pl < 0.05) || (pl > 0.95))		
			continue;
		//see if IV is less than 0.25		
		if(IV < 0.25)
			continue;			

    //calculate the vector for updating R
    #pragma omp parallel for
    for(j=0;j<INDIVIDUALS;j++)
    {
      if(SNP_[j] > 0.1)
  		{
        v_m[i*INDIVIDUALS+j] = 1;
			}	
      v_r[i*INDIVIDUALS+j] = v_m[i*INDIVIDUALS+j] * SNPFrequency_pl[j]/sqrt(2*pl*(1-pl));
    }
	}
  
  //variables for ssyr
  CBLAS_UPLO UPLOA = CblasLower;
  CBLAS_TRANSPOSE TRANS = CblasNoTrans;
  int N = INDIVIDUALS;
  int K = GENENUM;
  float ALPHA = 1;
  int LDA = INDIVIDUALS;
  int BETA = 1;
  int LDC = INDIVIDUALS;

  //update M
  cblas_ssyrk(CblasColMajor,UPLOA,TRANS,N,K,ALPHA,v_m,LDA,BETA,M,LDC);
  //update R
  cblas_ssyrk(CblasColMajor,UPLOA,TRANS,N,K,ALPHA,v_r,LDA,BETA,R,LDC); 

	//Divide R by M
  #pragma omp parallel for
	for(i=0;i<INDIVIDUALS;i++)
	{
		for(j=0;j<INDIVIDUALS;j++)
		{
			R[i*INDIVIDUALS+j] /= M[i*INDIVIDUALS+j];
		}
	}

  //print first 2 rows, first 10 columns
	for(i=0;i<2;i++)
	{
		for(j=0;j<10;j++)
		{
			printf("%10.7f ",R[i*INDIVIDUALS+j]);
		}
		printf("\n");
	}

  //delete
  delete []SNP;
  delete []SNP_;
  delete []SNPFrequency;
  delete []M;
  delete []R;

	fclose(pFile);

}