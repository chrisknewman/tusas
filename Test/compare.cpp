#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <typeinfo>
#include <string>
#include <cstring>
#include "../tusas.h"
using namespace std;

int main() {

  string exodiff_location = TRILINOS_DIR + "bin/exodiff";
  string exodiff_1_str = exodiff_location + " HeatQuad/results.e HeatQuad/Gold.e";
  string exodiff_2_str = exodiff_location + " HeatHex/results.e HeatHex/Gold.e";
  string exodiff_3_str = exodiff_location + " HeatHexNoPrec/results.e HeatHexNoPrec/Gold.e";
  string exodiff_4_str = exodiff_location + " HeatTet/results.e HeatTet/Gold.e";
  string exodiff_5_str = exodiff_location + " PhaseHeatQuadImp/results.e PhaseHeatQuadImp/Gold.e";
  string exodiff_6_str = exodiff_location + " PhaseHeatQuadExp/results.e PhaseHeatQuadExp/Gold.e";
  string exodiff_7_str = exodiff_location + " PhaseHeatQuad/results.e PhaseHeatQuad/Gold.e";
  string exodiff_8_str = exodiff_location + " PhaseHeatQuadNoPrec/results.e PhaseHeatQuadNoPrec/Gold.e";
  string exodiff_9_str = exodiff_location + " PhaseHeatTris/results.e PhaseHeatTris/Gold.e";
  string exodiff_10_str = exodiff_location + " PhaseHeatQuadPar/results.e PhaseHeatQuadPar/Gold.e";
  string exodiff_11_str = exodiff_location + " PhaseHeatQuadParNoPrec/results.e PhaseHeatQuadParNoPrec/Gold.e";

  // ------------------------------------- Amend here for new test ------------------------------------------------//


 char exodiff_1[900];
  strcpy(exodiff_1,exodiff_1_str.c_str());
  char exodiff_2[900];
  strcpy(exodiff_2,exodiff_2_str.c_str());
  char exodiff_3[900];
  strcpy(exodiff_3,exodiff_3_str.c_str());
  char exodiff_4[900];
  strcpy(exodiff_4,exodiff_4_str.c_str());
  char exodiff_5[900];
  strcpy(exodiff_5,exodiff_5_str.c_str());
  char exodiff_6[900];
  strcpy(exodiff_6,exodiff_6_str.c_str());
  char exodiff_7[900];
  strcpy(exodiff_7,exodiff_7_str.c_str());
  char exodiff_8[900];
  strcpy(exodiff_8,exodiff_8_str.c_str());
  char exodiff_9[900];
  strcpy(exodiff_9,exodiff_9_str.c_str());
  char exodiff_10[900];
  strcpy(exodiff_10,exodiff_10_str.c_str());
  char exodiff_11[900];
  strcpy(exodiff_11,exodiff_11_str.c_str());

  // ------------------------------------- Amend here for new test ------------------------------------------------//       

  

  // HEAT QUAD //
  string HeatQuad = "F";           // Set to "F" for fail. Will be changed if output and gold are 'same' by exodiff
  FILE *in;
  char temp[512];
  if(!(in = popen(exodiff_1, "r"))){ return 1;  }
  ofstream ofs;                                                      // Deleting old temporary file
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);  // "     "   "   "     "    "
  ofs.close();                                                       // "     "   "   "     "    " 
  while(fgets(temp, sizeof(temp), in)!=NULL){                          // Loop through and put comparison results
  ofstream ofs;                                                        // from exodiff into temp.txt
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);    //
  ofs << temp;
  ofs.close();}
  pclose(in);

  bool CheckWord(char* str);     // Check to see if the comparison 
  ifstream file("temp.txt");     // deemed the two files 'same'
  char aWord[50];
  while (file.good()) {
    file>>aWord;
    if (file.good() && strcmp(aWord,"same" ) == 0) {
      //TRUE
                                   // If they are the same, change
      HeatQuad = "P";              // test_1 to "P" for pass
    }
  }
  // FALSE                       // If not, then it will remain "F"
  


  // HEAT HEX //
  string HeatHex = "F";
  char temp_2[512];
  if(!(in = popen(exodiff_2, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();                                                      
  while(fgets(temp, sizeof(temp), in)!=NULL){                      
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);    
  ifstream file_2("temp.txt");  
  char aWord_2[50];
  while (file_2.good()) {
    file_2>>aWord_2;
    if (file_2.good() && strcmp(aWord_2,"same" ) == 0) {
    HeatHex = "P";    
    }
  }


  // HEAT HEX NO PRECONDITIONER //                   
  string HeatHexNoPrec = "F";      
  char temp_3[512];
  if(!(in = popen(exodiff_3, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){                      
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);  
  ifstream file_3("temp.txt");
  char aWord_3[50];
  while (file_3.good()) {
    file_3>>aWord_3;
    if (file_3.good() && strcmp(aWord_3,"same" ) == 0) {
      HeatHexNoPrec = "P"; 
    }
  }


  // HEAT TET //
   string HeatTet = "F";
  char temp_4[512];
  if(!(in = popen(exodiff_4, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);
  ifstream file_4("temp.txt");
  char aWord_4[50];
  while (file_4.good()) {
    file_4>>aWord_4;
    if (file_4.good() && strcmp(aWord_4,"same" ) == 0) {
    HeatTet = "P";
    }
  }

  // PHASE HEAT QUAD //  
  string PhaseHeatQuadImp = "F";
  char temp_5[512];
  if(!(in = popen(exodiff_5, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);
  ifstream file_5("temp.txt");
  char aWord_5[50];
  while (file_5.good()) {
    file_5>>aWord_5;
    if (file_5.good() && strcmp(aWord_5,"same" ) == 0) {
    PhaseHeatQuadImp = "P";
    }
  }

  // PHASE HEAT QUAD EXPLICIT //  
  string PhaseHeatQuadExp = "F";
  char temp_6[512];
  if(!(in = popen(exodiff_6, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);
  ifstream file_6("temp.txt");
  char aWord_6[50];
  while (file_6.good()) {
    file_6>>aWord_6;
    if (file_6.good() && strcmp(aWord_6,"same" ) == 0) {
    PhaseHeatQuadExp = "P";
    }
  }

  // PHASE HEAT QUAD //  
  string PhaseHeatQuad = "F";
  char temp_7[512];
  if(!(in = popen(exodiff_7, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);
  ifstream file_7("temp.txt");
  char aWord_7[50];
  while (file_7.good()) {
    file_7>>aWord_7;
    if (file_7.good() && strcmp(aWord_7,"same" ) == 0) {
    PhaseHeatQuad = "P";
    }
  }


  // PHASE HEAT QUAD NO PRECONDITIONER //   
  string PhaseHeatQuadNoPrec = "F";
  char temp_8[512];
  if(!(in = popen(exodiff_8, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);
  ifstream file_8("temp.txt");
  char aWord_8[50];
  while (file_8.good()) {
    file_8>>aWord_8;
    if (file_8.good() && strcmp(aWord_8,"same" ) == 0) {
    PhaseHeatQuadNoPrec = "P";
    }
  }

  // PHASE HEAT TRIS //            
  string PhaseHeatTris = "F";
  char temp_9[512];
  if(!(in = popen(exodiff_9, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);
  ifstream file_9("temp.txt");
  char aWord_9[50];
  while (file_9.good()) {
    file_9>>aWord_9;
    if (file_9.good() && strcmp(aWord_9,"same" ) == 0) {
    PhaseHeatTris = "P";
    }
  }

  // PHASE HEAT QUAD IN PARALLEL //
  string PhaseHeatQuadPar = "F";
  char temp_10[512];
  if(!(in = popen(exodiff_10, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);
  ifstream file_10("temp.txt");
  char aWord_10[50];
  while (file_10.good()) {
    file_10>>aWord_10;
    if (file_10.good() && strcmp(aWord_10,"same" ) == 0) {
      PhaseHeatQuadPar = "P";
    }
  }

  // PHASE HEAT QUAD NO PRECONDITIONER IN PARALLEL // 
  string PhaseHeatQuadParNoPrec = "F";
  char temp_11[512];
  if(!(in = popen(exodiff_11, "r"))){ return 1;  }
  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  ofs.close();
  while(fgets(temp, sizeof(temp), in)!=NULL){
    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
    ofs << temp;
    ofs.close();}
  pclose(in);

  bool CheckWord(char* str);
  ifstream file_11("temp.txt");
  char aWord_11[50];
  while (file_11.good()) {
    file_11>>aWord_11;
    if (file_11.good() && strcmp(aWord_11,"same" ) == 0) {
      PhaseHeatQuadParNoPrec = "P";
    }
  }




  // ------------------------------------- Amend here above for new test -----------------------------------------//       
  // ------------------------------------------ Copy a test and amend  -------------------------------------------//       

  // -------AMEND TO DESCRIBE NEW TEST------- //
  //  string AMEND-TO-BE-TEST-NAME = "F";  <----------------------------------- Change to test name
  //  char temp_1000[512];                 <---------------------------------Change 1000 to n if this is the nth test
  //  if(!(in = popen(exodiff_1000, "r"))){ return 1;  }    <----------------Change 1000 to n if this is the nth test      
  //  ofs.open ("temp.txt", std::ofstream::out | std::ofstream::trunc);
  //  ofs.close();
  //  while(fgets(temp, sizeof(temp), in)!=NULL){
  //    ofs.open ("temp.txt", std::ofstream::out | std::ofstream::app);
  //    ofs << temp;
  //    ofs.close();}
  //  pclose(in);

  //  bool CheckWord(char* str);
  //  ifstream file_1000("temp.txt");          <------------------------------Change 1000 to n if this is the nth test      
  //  char aWord_1000[50];                     <------------------------------Change 1000 to n if this is the nth test      
  //  while (file_1000.good()) {               <------------------------------Change 1000 to n if this is the nth test      
  //    file_1000>>aWord_1000;                 <------------------------------Change 1000 to n if this is the nth test      
  //    if (file_1000.good() && strcmp(aWord_1000,"same" ) == 0) {  <---------Change 1000 to n if this is the nth test      
  //      AMEND-TO-BE-TEST-NAME = "P";         <---------------------------------- Change to test name        
  //    }
  //  }

  system ("rm temp.txt");
   cout << " HeatQuad: "<< HeatQuad << endl
	<< " HeatHex: "<< HeatHex << endl
	<< " HeatHexNoPrec: " << HeatHexNoPrec << endl
	<< " HeatTet: " << HeatTet << endl
        << " PhaseHeatQuadImp: " << PhaseHeatQuadImp << endl
        << " PhaseHeatQuadExp: " << PhaseHeatQuadExp << endl
        << " PhaseHeatQuad: " << PhaseHeatQuad << endl
        << " PhaseHeatQuadNoPrec: " << PhaseHeatQuadNoPrec << endl
        << " PhaseHeatTris: " << PhaseHeatTris << endl
	<< " PhaseHeatQuadPar: " << PhaseHeatQuadPar << endl
        << " PhaseHeatQuadParNoPrec: " << PhaseHeatQuadParNoPrec << endl

     // ------------------------------------- Amend here for new test ------------------------------------------------//       

	<< "\n";
   

  return 0;
}

