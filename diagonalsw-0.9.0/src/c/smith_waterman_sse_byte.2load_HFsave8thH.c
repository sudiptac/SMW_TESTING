

#include <tmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>  
#include <pmmintrin.h>

#include <stdio.h>

#include "sw_constants.h"
#include "smith_waterman_vector.h"
#include "sse_funcs.h"





int
 smith_waterman_vector_byte_H
(unsigned char *     query_sequence,
                            unsigned char *     query_profile_byte,
                            int                 query_length,
                            unsigned char *     db_sequence,
                            int                 db_length,
                            unsigned char       bias,
                            unsigned char       gap_open,
                            unsigned char       gap_extend,
                            unsigned char *     workspace

,         unsigned short *    Hmatrix


 )
{



    int                     i,j,k,k8;
    int                     overflow;
    unsigned char *         p;
    unsigned char           score;   

    __m128i     Fup,Hup1,Hup2,E,F,H,tmp;
    __m128i    v_maxscore;
    __m128i    v_bias,v_gapopen,v_gapextend;
    __m128i    v_score;


    __m128i    v_score_load1;
    __m128i    v_score_load2;  

   __m128i     HFload,HFsave;

    __m128i v0;
    __m128i v1;
    __m128i v2;
    __m128i v3;
    __m128i v4;
    __m128i v5;
    __m128i v6;
    __m128i v7;


    const  __m128i mask1=_mm_set_epi8(0,0,0,0,255,255,255,255,0,0,0,0,255,255,255,255);
    const  __m128i mask2=_mm_set_epi8(0,0,255,255,0,0,255,255,0,0,255,255,0,0,255,255);
    const  __m128i mask3=_mm_set_epi8(0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255);




      
    /* Load the bias to all elements of a constant */
    v_bias = _mm_set1_epi8(bias);
    
    /* Load gap opening penalty to all elements of a constant */

    v_gapopen = _mm_set1_epi8(gap_open);

    /* Load gap extension penalty to all elements of a constant */

    v_gapextend = _mm_set1_epi8(gap_extend);

    v_maxscore = _mm_setzero_si128();

    // Zero out the storage vector 
    k = (db_length+15)/8;

    __m128i * iter_ptr;
    for(i=0, iter_ptr =  ( __m128i * ) workspace  ;__builtin_expect((i<k),1);i++,iter_ptr++)
    {
        // borrow the zero value in v_maxscore to have something to store
      _mm_store_si128( (__m128i *) iter_ptr , v_maxscore);
    }



    for(i=0;i<query_length;i+=16)
    {
        int dbindex=0;
        v0=_mm_set1_epi8(0);
        v1=_mm_set1_epi8(0);
        v2=_mm_set1_epi8(0);
        v3=_mm_set1_epi8(0);
        v4=_mm_set1_epi8(0);
        v5=_mm_set1_epi8(0);
        v6=_mm_set1_epi8(0);
        v7=_mm_set1_epi8(0);


        E          = _mm_setzero_si128();
        F          = _mm_setzero_si128();
        H          = _mm_setzero_si128();
        Hup2      = _mm_setzero_si128();


        // reset pointers to the start of the saved data from the last row
        p = workspace;
        
        // start directly and prefetch score column
        k             = db_sequence[0];
        k8            = k;

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) );

        v_score_load2 = v_score_load1;

        v_score_load1 = _mm_srli_si128(v_score_load1,8);



  



  
    k = db_sequence[1];
  
  
  
  

  v7 = v_score_load1;
  v3=_mm_blendv_epi8(v3,v7,mask1);
  v1=_mm_blendv_epi8(v1,v3,mask2);
  v0=_mm_blendv_epi8(v0,v1,mask3);

  v_score = v0;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
  


  

  

  

  

  

   HFload  =  _mm_load_si128( (__m128i *) p );  
  
  
  

  
  
  
  
  

   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[2];
  
  
  
  

  v0 = v_score_load1;
  v4=_mm_blendv_epi8(v4,v0,mask1);
  v2=_mm_blendv_epi8(v2,v4,mask2);
  v1=_mm_blendv_epi8(v1,v2,mask3);

  v_score = v1;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
  


  

  

  

  

  

  
  
  
  

  
  
  
  
  

   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[3];
  
  
  
  

  v1 = v_score_load1;
  v5=_mm_blendv_epi8(v5,v1,mask1);
  v3=_mm_blendv_epi8(v3,v5,mask2);
  v2=_mm_blendv_epi8(v2,v3,mask3);

  v_score = v2;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
  


  

  

  

  

  

  
  
  
  

  
  
  
  
  

   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[4];
  
  
  
  

  v2 = v_score_load1;
  v6=_mm_blendv_epi8(v6,v2,mask1);
  v4=_mm_blendv_epi8(v4,v6,mask2);
  v3=_mm_blendv_epi8(v3,v4,mask3);

  v_score = v3;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
  


  

  

  

  

  

  
  
  
  

  
  
  
  
  

   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 
  



  
    k = db_sequence[5];
  
  
  
  

  v3 = v_score_load1;
  v7=_mm_blendv_epi8(v7,v3,mask1);
  v5=_mm_blendv_epi8(v5,v7,mask2);
  v4=_mm_blendv_epi8(v4,v5,mask3);

  v_score = v4;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
  


  

  

  

  

  

  
  
  
  

  
  
  
  
  

   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[6];
  
  
  
  

  v4 = v_score_load1;
  v0=_mm_blendv_epi8(v0,v4,mask1);
  v6=_mm_blendv_epi8(v6,v0,mask2);
  v5=_mm_blendv_epi8(v5,v6,mask3);

  v_score = v5;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
  


  

  

  

  

  

  
  
  
  

  
  
  
  
  

   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[7];
  
  
  
  

  v5 = v_score_load1;
  v1=_mm_blendv_epi8(v1,v5,mask1);
  v7=_mm_blendv_epi8(v7,v1,mask2);
  v6=_mm_blendv_epi8(v6,v7,mask3);

  v_score = v6;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
  


  

  

  

  

  

  
  
  
  

  
  
  
  
  

   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[8];
  
  
  
  

  v6 = v_score_load1;
  v2=_mm_blendv_epi8(v2,v6,mask1);
  v0=_mm_blendv_epi8(v0,v2,mask2);
  v7=_mm_blendv_epi8(v7,v0,mask3);

  v_score = v7;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
  


  

  

  

  

  

  
  
  
  

     p += 16; 
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 
  



  
    k = db_sequence[9];
  
   k8 = db_sequence[1];
  
  

  v7 = v_score_load1;
  v3=_mm_blendv_epi8(v3,v7,mask1);
  v1=_mm_blendv_epi8(v1,v3,mask2);
  v0=_mm_blendv_epi8(v0,v1,mask3);

  v_score = v0;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  

  

  

  

  
   HFload  =  _mm_load_si128( (__m128i *) p ); 
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[10];
  
   k8 = db_sequence[2];
  
  

  v0 = v_score_load1;
  v4=_mm_blendv_epi8(v4,v0,mask1);
  v2=_mm_blendv_epi8(v2,v4,mask2);
  v1=_mm_blendv_epi8(v1,v2,mask3);

  v_score = v1;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[11];
  
   k8 = db_sequence[3];
  
  

  v1 = v_score_load1;
  v5=_mm_blendv_epi8(v5,v1,mask1);
  v3=_mm_blendv_epi8(v3,v5,mask2);
  v2=_mm_blendv_epi8(v2,v3,mask3);

  v_score = v2;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[12];
  
   k8 = db_sequence[4];
  
  

  v2 = v_score_load1;
  v6=_mm_blendv_epi8(v6,v2,mask1);
  v4=_mm_blendv_epi8(v4,v6,mask2);
  v3=_mm_blendv_epi8(v3,v4,mask3);

  v_score = v3;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 
  



  
    k = db_sequence[13];
  
   k8 = db_sequence[5];
  
  

  v3 = v_score_load1;
  v7=_mm_blendv_epi8(v7,v3,mask1);
  v5=_mm_blendv_epi8(v5,v7,mask2);
  v4=_mm_blendv_epi8(v4,v5,mask3);

  v_score = v4;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[14];
  
   k8 = db_sequence[6];
  
  

  v4 = v_score_load1;
  v0=_mm_blendv_epi8(v0,v4,mask1);
  v6=_mm_blendv_epi8(v6,v0,mask2);
  v5=_mm_blendv_epi8(v5,v6,mask3);

  v_score = v5;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[15];
  
   k8 = db_sequence[7];
  
  

  v5 = v_score_load1;
  v1=_mm_blendv_epi8(v1,v5,mask1);
  v7=_mm_blendv_epi8(v7,v1,mask2);
  v6=_mm_blendv_epi8(v6,v7,mask3);

  v_score = v6;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
    k = db_sequence[16];
  
   k8 = db_sequence[8];
  
  

  v6 = v_score_load1;
  v2=_mm_blendv_epi8(v2,v6,mask1);
  v0=_mm_blendv_epi8(v0,v2,mask2);
  v7=_mm_blendv_epi8(v7,v0,mask3);

  v_score = v7;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  

  

  

  

  
  
  
  

  
   p += 16; 
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 





//printf("inner loop ---------------------------\n");        
        for(j=16;__builtin_expect(( j<db_length ),1);j+=16)
        { 


  



  
  
   k = db_sequence[j+1];
  
  k8 = db_sequence[j+(-7)];
  

  v7 = v_score_load1;
  v3=_mm_blendv_epi8(v3,v7,mask1);
  v1=_mm_blendv_epi8(v1,v3,mask2);
  v0=_mm_blendv_epi8(v0,v1,mask3);

  v_score = v0;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
   HFload  =  _mm_load_si128( (__m128i *) p ); 
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+2];
  
  k8 = db_sequence[j+(-6)];
  

  v0 = v_score_load1;
  v4=_mm_blendv_epi8(v4,v0,mask1);
  v2=_mm_blendv_epi8(v2,v4,mask2);
  v1=_mm_blendv_epi8(v1,v2,mask3);

  v_score = v1;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+3];
  
  k8 = db_sequence[j+(-5)];
  

  v1 = v_score_load1;
  v5=_mm_blendv_epi8(v5,v1,mask1);
  v3=_mm_blendv_epi8(v3,v5,mask2);
  v2=_mm_blendv_epi8(v2,v3,mask3);

  v_score = v2;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+4];
  
  k8 = db_sequence[j+(-4)];
  

  v2 = v_score_load1;
  v6=_mm_blendv_epi8(v6,v2,mask1);
  v4=_mm_blendv_epi8(v4,v6,mask2);
  v3=_mm_blendv_epi8(v3,v4,mask3);

  v_score = v3;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 
  



  
  
   k = db_sequence[j+5];
  
  k8 = db_sequence[j+(-3)];
  

  v3 = v_score_load1;
  v7=_mm_blendv_epi8(v7,v3,mask1);
  v5=_mm_blendv_epi8(v5,v7,mask2);
  v4=_mm_blendv_epi8(v4,v5,mask3);

  v_score = v4;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+6];
  
  k8 = db_sequence[j+(-2)];
  

  v4 = v_score_load1;
  v0=_mm_blendv_epi8(v0,v4,mask1);
  v6=_mm_blendv_epi8(v6,v0,mask2);
  v5=_mm_blendv_epi8(v5,v6,mask3);

  v_score = v5;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+7];
  
  k8 = db_sequence[j+(-1)];
  

  v5 = v_score_load1;
  v1=_mm_blendv_epi8(v1,v5,mask1);
  v7=_mm_blendv_epi8(v7,v1,mask2);
  v6=_mm_blendv_epi8(v6,v7,mask3);

  v_score = v6;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+8];
  
  k8 = db_sequence[j+(0)];
  

  v6 = v_score_load1;
  v2=_mm_blendv_epi8(v2,v6,mask1);
  v0=_mm_blendv_epi8(v0,v2,mask2);
  v7=_mm_blendv_epi8(v7,v0,mask3);

  v_score = v7;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  
   _mm_store_si128( (__m128i *) (  p - 32 ), HFsave);
   HFsave = _mm_setzero_si128();
  

  

  

  
  
  
  

  
  
   p += 16; 
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 
  



  
  
   k = db_sequence[j+9];
  
  k8 = db_sequence[j+(1)];
  

  v7 = v_score_load1;
  v3=_mm_blendv_epi8(v3,v7,mask1);
  v1=_mm_blendv_epi8(v1,v3,mask2);
  v0=_mm_blendv_epi8(v0,v1,mask3);

  v_score = v0;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
   HFload  =  _mm_load_si128( (__m128i *) p ); 

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+10];
  
  k8 = db_sequence[j+(2)];
  

  v0 = v_score_load1;
  v4=_mm_blendv_epi8(v4,v0,mask1);
  v2=_mm_blendv_epi8(v2,v4,mask2);
  v1=_mm_blendv_epi8(v1,v2,mask3);

  v_score = v1;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+11];
  
  k8 = db_sequence[j+(3)];
  

  v1 = v_score_load1;
  v5=_mm_blendv_epi8(v5,v1,mask1);
  v3=_mm_blendv_epi8(v3,v5,mask2);
  v2=_mm_blendv_epi8(v2,v3,mask3);

  v_score = v2;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+12];
  
  k8 = db_sequence[j+(4)];
  

  v2 = v_score_load1;
  v6=_mm_blendv_epi8(v6,v2,mask1);
  v4=_mm_blendv_epi8(v4,v6,mask2);
  v3=_mm_blendv_epi8(v3,v4,mask3);

  v_score = v3;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 
  



  
  
   k = db_sequence[j+13];
  
  k8 = db_sequence[j+(5)];
  

  v3 = v_score_load1;
  v7=_mm_blendv_epi8(v7,v3,mask1);
  v5=_mm_blendv_epi8(v5,v7,mask2);
  v4=_mm_blendv_epi8(v4,v5,mask3);

  v_score = v4;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+14];
  
  k8 = db_sequence[j+(6)];
  

  v4 = v_score_load1;
  v0=_mm_blendv_epi8(v0,v4,mask1);
  v6=_mm_blendv_epi8(v6,v0,mask2);
  v5=_mm_blendv_epi8(v5,v6,mask3);

  v_score = v5;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+15];
  
  k8 = db_sequence[j+(7)];
  

  v5 = v_score_load1;
  v1=_mm_blendv_epi8(v1,v5,mask1);
  v7=_mm_blendv_epi8(v7,v1,mask2);
  v6=_mm_blendv_epi8(v6,v7,mask3);

  v_score = v6;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup1    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 



  
  
   k = db_sequence[j+16];
  
  k8 = db_sequence[j+(8)];
  

  v6 = v_score_load1;
  v2=_mm_blendv_epi8(v2,v6,mask1);
  v0=_mm_blendv_epi8(v0,v2,mask2);
  v7=_mm_blendv_epi8(v7,v0,mask3);

  v_score = v7;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

   v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k ) ); 
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


  

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  
   _mm_store_si128( (__m128i *) ( p -32 ), HFsave);
   HFsave = _mm_setzero_si128();
  

  

  
  
  
  

  
  
  
   p += 16; 
  

  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 

  

  
   // shift into place so we have complete F and H vectors
   // that refer to the values one unit up from each cell
   // that we are currently working on.
   Fup = _mm_alignr_epi8(F,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
  

  
   Hup2    = _mm_alignr_epi8(H,HFload,15);
   HFload=_mm_slli_si128(HFload,1);
   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
 

        }
        
//printf("end loop ---------------------------\n");        

  



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v7 = v_score_load1;
  v3=_mm_blendv_epi8(v3,v7,mask1);
  v1=_mm_blendv_epi8(v1,v3,mask2);
  v0=_mm_blendv_epi8(v0,v1,mask3);

  v_score = v0;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup1   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v0 = v_score_load1;
  v4=_mm_blendv_epi8(v4,v0,mask1);
  v2=_mm_blendv_epi8(v2,v4,mask2);
  v1=_mm_blendv_epi8(v1,v2,mask3);

  v_score = v1;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup2   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v1 = v_score_load1;
  v5=_mm_blendv_epi8(v5,v1,mask1);
  v3=_mm_blendv_epi8(v3,v5,mask2);
  v2=_mm_blendv_epi8(v2,v3,mask3);

  v_score = v2;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup1   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v2 = v_score_load1;
  v6=_mm_blendv_epi8(v6,v2,mask1);
  v4=_mm_blendv_epi8(v4,v6,mask2);
  v3=_mm_blendv_epi8(v3,v4,mask3);

  v_score = v3;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup2   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 
  



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v3 = v_score_load1;
  v7=_mm_blendv_epi8(v7,v3,mask1);
  v5=_mm_blendv_epi8(v5,v7,mask2);
  v4=_mm_blendv_epi8(v4,v5,mask3);

  v_score = v4;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup1   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v4 = v_score_load1;
  v0=_mm_blendv_epi8(v0,v4,mask1);
  v6=_mm_blendv_epi8(v6,v0,mask2);
  v5=_mm_blendv_epi8(v5,v6,mask3);

  v_score = v5;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup2   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v5 = v_score_load1;
  v1=_mm_blendv_epi8(v1,v5,mask1);
  v7=_mm_blendv_epi8(v7,v1,mask2);
  v6=_mm_blendv_epi8(v6,v7,mask3);

  v_score = v6;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup1   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v6 = v_score_load1;
  v2=_mm_blendv_epi8(v2,v6,mask1);
  v0=_mm_blendv_epi8(v0,v2,mask2);
  v7=_mm_blendv_epi8(v7,v0,mask3);

  v_score = v7;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  
   _mm_store_si128( (__m128i *) ( p - 32 ) , HFsave);
   HFsave = _mm_setzero_si128();
  

  
  
  
  

  
  
  
  
   p += 16; 

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup2   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 
  



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v7 = v_score_load1;
  v3=_mm_blendv_epi8(v3,v7,mask1);
  v1=_mm_blendv_epi8(v1,v3,mask2);
  v0=_mm_blendv_epi8(v0,v1,mask3);

  v_score = v0;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup1   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v0 = v_score_load1;
  v4=_mm_blendv_epi8(v4,v0,mask1);
  v2=_mm_blendv_epi8(v2,v4,mask2);
  v1=_mm_blendv_epi8(v1,v2,mask3);

  v_score = v1;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup2   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v1 = v_score_load1;
  v5=_mm_blendv_epi8(v5,v1,mask1);
  v3=_mm_blendv_epi8(v3,v5,mask2);
  v2=_mm_blendv_epi8(v2,v3,mask3);

  v_score = v2;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup1   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v2 = v_score_load1;
  v6=_mm_blendv_epi8(v6,v2,mask1);
  v4=_mm_blendv_epi8(v4,v6,mask2);
  v3=_mm_blendv_epi8(v3,v4,mask3);

  v_score = v3;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup2   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 
  



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v3 = v_score_load1;
  v7=_mm_blendv_epi8(v7,v3,mask1);
  v5=_mm_blendv_epi8(v5,v7,mask2);
  v4=_mm_blendv_epi8(v4,v5,mask3);

  v_score = v4;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup1   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v4 = v_score_load1;
  v0=_mm_blendv_epi8(v0,v4,mask1);
  v6=_mm_blendv_epi8(v6,v0,mask2);
  v5=_mm_blendv_epi8(v5,v6,mask3);

  v_score = v5;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup2   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup2,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup1,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 



  
   if (j >= db_length+15 ) goto ending;
  
  
  
  
  
  k8 = db_sequence[j-7];

  v5 = v_score_load1;
  v1=_mm_blendv_epi8(v1,v5,mask1);
  v7=_mm_blendv_epi8(v7,v1,mask2);
  v6=_mm_blendv_epi8(v6,v7,mask3);

  v_score = v6;

  E = _mm_max_epu8( _mm_subs_epu8(E,v_gapextend),_mm_subs_epu8(H,v_gapopen));

  
   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_byte + 16*k8 ) ); 


   v_score_load1 = _mm_slli_si128(v_score_load2,8); 

  
   HFsave=_mm_alignr_epi8(HFsave,F,15);
   HFsave=_mm_alignr_epi8(HFsave,H,15);
  

  

  

  

  
  
  
  

  
  
  
  
  

  
  

  
   Fup = _mm_slli_si128(F,1);
   Hup1   = _mm_slli_si128(H,1); 
  

  

   

  F   = _mm_subs_epu8(Fup,v_gapextend);
  tmp = _mm_subs_epu8(Hup1,v_gapopen);
  F   = _mm_max_epu8(F,tmp);
  H   = _mm_subs_epu8( _mm_adds_epu8(Hup2,v_score),v_bias);
  H   = _mm_max_epu8(H,E);
  H   = _mm_max_epu8(H,F);





 set_matrix_values_from_diagonal_byte_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;



  v_maxscore = _mm_max_epu8(v_maxscore,H);


  
   j++;  
  
 




ending:

  HFsave=_mm_alignr_epi8(HFsave,F,15);
  HFsave=_mm_alignr_epi8(HFsave,H,15);


for ( j = 0 ; j <   7 - (( db_length -1 ) % 8 ) ; j++ )
{
   HFsave = _mm_slli_si128(HFsave, 2  );
}


       _mm_store_si128( (__m128i *) ( p -32 ), HFsave);

        query_profile_byte += 16*ALPHABET_SIZE;

        // End of this row (actually 16 rows due to SIMD).
        // Before we continue, check for overflow.

        tmp      = _mm_subs_epu8(_mm_set1_epi8(255),v_bias);
        tmp      = _mm_cmpeq_epi8 (tmp,v_maxscore);
        int noOverflow =  _mm_testc_si128( _mm_setzero_si128(),tmp);
        overflow = ! noOverflow ;


    }

    if(__builtin_expect (overflow, 0))
    {
        return -1;
    }
    else
    {
        // find largest score in the v_maxscore vector

        tmp = _mm_alignr_epi8(v_maxscore,v_maxscore,8);
        v_maxscore = _mm_max_epu8(v_maxscore,tmp);
        tmp = _mm_alignr_epi8(v_maxscore,v_maxscore,4);
        v_maxscore = _mm_max_epu8(v_maxscore,tmp);
        tmp = _mm_alignr_epi8(v_maxscore,v_maxscore,2);
        v_maxscore = _mm_max_epu8(v_maxscore,tmp);
        tmp = _mm_alignr_epi8(v_maxscore,v_maxscore,1);
        v_maxscore = _mm_max_epu8(v_maxscore,tmp);


        // store in temporary variable
        score=_mm_extract_epi8(v_maxscore,0);
      
        // return largest score

        return score;
    }
}



 