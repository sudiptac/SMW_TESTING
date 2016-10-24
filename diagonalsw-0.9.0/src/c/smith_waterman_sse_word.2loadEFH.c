

#include <stdio.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>  
#include <pmmintrin.h>


#include "sw_constants.h"
#include "smith_waterman_vector.h"
#include "sse_funcs.h"





int
smith_waterman_vector_word_EFH
(unsigned char *     query_sequence,
                            unsigned short *    query_profile_word,
                            int                 query_length,
                            unsigned char *     db_sequence,
                            int                 db_length,
                            unsigned short      bias,
                            unsigned short      gap_open,
                            unsigned short      gap_extend,
                            unsigned short *    workspace

,       unsigned short *    Ematrix,
			    unsigned short *    Fmatrix,
			    unsigned short *    Hmatrix

)
{
    int                     i,j,k,k2;
    unsigned short *        p;
    unsigned short          score;   
    unsigned char *         p_dbseq;
    
    __m128i   Fup,Hup1,Hup2,E,F,H,tmp;
    __m128i    perm;
    __m128i   v_maxscore;
    __m128i   v_bias,v_gapopen,v_gapextend;
    __m128i   v_score;
    __m128i   v_score_load2; 
    __m128i   v_score_load1; 


//    const  __m128i mask1=_mm_set_epi8(0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255);
    const  __m128i mask2=_mm_set_epi8(0,0,0,0,255,255,255,255,0,0,0,0,255,255,255,255);
    const  __m128i mask3=_mm_set_epi8(0,0,255,255,0,0,255,255,0,0,255,255,0,0,255,255);



      
    /* Load the bias to all elements of a constant */
    v_bias = _mm_set1_epi16(bias);  

  
    /* Load gap opening penalty to all elements of a constant */
    v_gapopen = _mm_set1_epi16(gap_open);  

    /* Load gap extension penalty to all elements of a constant */
    v_gapextend = _mm_set1_epi16(gap_extend);  

    v_maxscore = _mm_setzero_si128();
   
    // Zero out the storage vector 
    k = 2*(db_length+7);
        
    __m128i * iter_ptr;
    for(i=0, iter_ptr =  ( __m128i * ) workspace  ;i<k;i++,iter_ptr++)
    {
        // borrow the zero value in v_maxscore to have something to store

      /*
        vec_st(v_maxscore,j,workspace);
      */
      _mm_store_si128( (__m128i *) iter_ptr ,v_maxscore);
    }

    
    for(i=0;i<query_length;i+=8)
    {
   int dbindex=0;

        p_dbseq    = db_sequence;
        k          = *p_dbseq++;



    __m128i v0=_mm_set1_epi8(0);
    __m128i v1=_mm_set1_epi8(0);
    __m128i v2=_mm_set1_epi8(0);
  __m128i v3;
//    __m128i v3=_mm_set1_epi8(0);


/*
    __m128i v4=_mm_set1_epi8(0);
    __m128i v5=_mm_set1_epi8(0);
    __m128i v6=_mm_set1_epi8(0);
    __m128i v7=_mm_set1_epi8(0);

*/


        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );

        v_score_load2 = v_score_load1;
        v_score_load1 = _mm_srli_si128(v_score_load1,8);

        // zero lots of stuff. 
        // We use both the VPERM and VSIU unit to knock off some cycles.


        E          = _mm_setzero_si128();
        F          = _mm_setzero_si128();
        H          = _mm_setzero_si128();
        Hup2      = _mm_setzero_si128();

        // reset pointers to the start of the saved data from the last row
        p = workspace;








        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v3 = v_score_load1;


//       v3=_mm_blendv_epi8(v3,v3,mask1);
       v1=_mm_blendv_epi8(v1,v3,mask2);
       v0=_mm_blendv_epi8(v0,v1,mask3);

v_score     = v0;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



  

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup1   =  _mm_load_si128( (__m128i *)  (p+8) ); 



   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup1   = _mm_alignr_epi8(H,Hup1,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v0 = v_score_load1;


//       v4=_mm_blendv_epi8(v0,v0,mask1);
       v2=_mm_blendv_epi8(v2,v0,mask2);
       v1=_mm_blendv_epi8(v1,v2,mask3);

v_score     = v1;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



  

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup2   =  _mm_load_si128( (__m128i *)  (p+8) ); 



   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup2   = _mm_alignr_epi8(H,Hup2,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v1 = v_score_load1;


//       v5=_mm_blendv_epi8(v1,v1,mask1);
       v3=_mm_blendv_epi8(v3,v1,mask2);
       v2=_mm_blendv_epi8(v2,v3,mask3);

v_score     = v2;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



  

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup1   =  _mm_load_si128( (__m128i *)  (p+8) ); 



   v_score_load1 = _mm_srli_si128(v_score_load1,8); 
  


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup1   = _mm_alignr_epi8(H,Hup1,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v2 = v_score_load1;


//       v6=_mm_blendv_epi8(v2,v2,mask1);
       v0=_mm_blendv_epi8(v0,v2,mask2);
       v3=_mm_blendv_epi8(v3,v0,mask3);

v_score     = v3;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



  

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup2   =  _mm_load_si128( (__m128i *)  (p+8) ); 



  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup2   = _mm_alignr_epi8(H,Hup2,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);










        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v3 = v_score_load1;


//       v7=_mm_blendv_epi8(v3,v3,mask1);
       v1=_mm_blendv_epi8(v1,v3,mask2);
       v0=_mm_blendv_epi8(v0,v1,mask3);

v_score     = v0;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup1   =  _mm_load_si128( (__m128i *)  (p+8) ); 



  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup1   = _mm_alignr_epi8(H,Hup1,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v0 = v_score_load1;


//       v0=_mm_blendv_epi8(v0,v0,mask1);
       v2=_mm_blendv_epi8(v2,v0,mask2);
       v1=_mm_blendv_epi8(v1,v2,mask3);

v_score     = v1;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup2   =  _mm_load_si128( (__m128i *)  (p+8) ); 



  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup2   = _mm_alignr_epi8(H,Hup2,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v1 = v_score_load1;


//       v1=_mm_blendv_epi8(v1,v1,mask1);
       v3=_mm_blendv_epi8(v3,v1,mask2);
       v2=_mm_blendv_epi8(v2,v3,mask3);

v_score     = v2;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup1   =  _mm_load_si128( (__m128i *)  (p+8) ); 



  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup1   = _mm_alignr_epi8(H,Hup1,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v2 = v_score_load1;


//       v2=_mm_blendv_epi8(v2,v2,mask1);
       v0=_mm_blendv_epi8(v0,v2,mask2);
       v3=_mm_blendv_epi8(v3,v0,mask3);

v_score     = v3;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup2   =  _mm_load_si128( (__m128i *)  (p+8) ); 



  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup2   = _mm_alignr_epi8(H,Hup2,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);






   

        for(j=8;j<db_length;j+=8)
        {           








        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v3 = v_score_load1;


//       v3=_mm_blendv_epi8(v3,v3,mask1);
       v1=_mm_blendv_epi8(v1,v3,mask2);
       v0=_mm_blendv_epi8(v0,v1,mask3);

v_score     = v0;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup1   =  _mm_load_si128( (__m128i *)  (p+8) ); 


            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup1   = _mm_alignr_epi8(H,Hup1,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v0 = v_score_load1;


//       v4=_mm_blendv_epi8(v0,v0,mask1);
       v2=_mm_blendv_epi8(v2,v0,mask2);
       v1=_mm_blendv_epi8(v1,v2,mask3);

v_score     = v1;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup2   =  _mm_load_si128( (__m128i *)  (p+8) ); 


            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup2   = _mm_alignr_epi8(H,Hup2,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v1 = v_score_load1;


//       v5=_mm_blendv_epi8(v1,v1,mask1);
       v3=_mm_blendv_epi8(v3,v1,mask2);
       v2=_mm_blendv_epi8(v2,v3,mask3);

v_score     = v2;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup1   =  _mm_load_si128( (__m128i *)  (p+8) ); 


            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup1   = _mm_alignr_epi8(H,Hup1,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v2 = v_score_load1;


//       v6=_mm_blendv_epi8(v2,v2,mask1);
       v0=_mm_blendv_epi8(v0,v2,mask2);
       v3=_mm_blendv_epi8(v3,v0,mask3);

v_score     = v3;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup2   =  _mm_load_si128( (__m128i *)  (p+8) ); 


            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup2   = _mm_alignr_epi8(H,Hup2,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);










        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v3 = v_score_load1;


//       v7=_mm_blendv_epi8(v3,v3,mask1);
       v1=_mm_blendv_epi8(v1,v3,mask2);
       v0=_mm_blendv_epi8(v0,v1,mask3);

v_score     = v0;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup1   =  _mm_load_si128( (__m128i *)  (p+8) ); 


            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup1   = _mm_alignr_epi8(H,Hup1,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v0 = v_score_load1;


//       v0=_mm_blendv_epi8(v0,v0,mask1);
       v2=_mm_blendv_epi8(v2,v0,mask2);
       v1=_mm_blendv_epi8(v1,v2,mask3);

v_score     = v1;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup2   =  _mm_load_si128( (__m128i *)  (p+8) ); 


            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup2   = _mm_alignr_epi8(H,Hup2,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v1 = v_score_load1;


//       v1=_mm_blendv_epi8(v1,v1,mask1);
       v3=_mm_blendv_epi8(v3,v1,mask2);
       v2=_mm_blendv_epi8(v2,v3,mask3);

v_score     = v2;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup1   =  _mm_load_si128( (__m128i *)  (p+8) ); 


            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup1   = _mm_alignr_epi8(H,Hup1,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);









        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v2 = v_score_load1;


//       v2=_mm_blendv_epi8(v2,v2,mask1);
       v0=_mm_blendv_epi8(v0,v2,mask2);
       v3=_mm_blendv_epi8(v3,v0,mask3);

v_score     = v3;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));


        // prefetch score for next step 

        v_score_load1 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k ) );



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

  



        Fup    =  _mm_load_si128( (__m128i *)  p ); 
        Hup2   =  _mm_load_si128( (__m128i *)  (p+8) ); 


            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
   v_score_load1 = _mm_alignr_epi8( v_score_load2,v_score_load1,8); 


        p += 16; 


        // shift into place so we have complete F and H vectors
        // that refer to the values one unit up from each cell
        // that we are currently working on.
            Fup    = _mm_alignr_epi8(F,Fup,14);
        Hup2   = _mm_alignr_epi8(H,Hup2,14);



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);







        }






   if ( j >= db_length+7 ) goto ending;




        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v3 = v_score_load1;


//       v3=_mm_blendv_epi8(v3,v3,mask1);
       v1=_mm_blendv_epi8(v1,v3,mask2);
       v0=_mm_blendv_epi8(v0,v1,mask3);

v_score     = v0;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

   v_score_load1 = _mm_slli_si128(v_score_load2,8); 




            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
  


        p += 16; 



   
            Fup    =  _mm_slli_si128(F,2); 
            Hup1    = _mm_slli_si128(H,2); 



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);



        j++;  




   if ( j >= db_length+7 ) goto ending;




        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v0 = v_score_load1;


//       v4=_mm_blendv_epi8(v0,v0,mask1);
       v2=_mm_blendv_epi8(v2,v0,mask2);
       v1=_mm_blendv_epi8(v1,v2,mask3);

v_score     = v1;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

   v_score_load1 = _mm_slli_si128(v_score_load2,8); 




            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
  


        p += 16; 



   
            Fup    =  _mm_slli_si128(F,2); 
            Hup2    = _mm_slli_si128(H,2); 



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);



        j++;  




   if ( j >= db_length+7 ) goto ending;




        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v1 = v_score_load1;


//       v5=_mm_blendv_epi8(v1,v1,mask1);
       v3=_mm_blendv_epi8(v3,v1,mask2);
       v2=_mm_blendv_epi8(v2,v3,mask3);

v_score     = v2;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

   v_score_load1 = _mm_slli_si128(v_score_load2,8); 




            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
  


        p += 16; 



   
            Fup    =  _mm_slli_si128(F,2); 
            Hup1    = _mm_slli_si128(H,2); 



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);



        j++;  




   if ( j >= db_length+7 ) goto ending;




        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v2 = v_score_load1;


//       v6=_mm_blendv_epi8(v2,v2,mask1);
       v0=_mm_blendv_epi8(v0,v2,mask2);
       v3=_mm_blendv_epi8(v3,v0,mask3);

v_score     = v3;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

   v_score_load1 = _mm_slli_si128(v_score_load2,8); 




            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
  


        p += 16; 



   
            Fup    =  _mm_slli_si128(F,2); 
            Hup2    = _mm_slli_si128(H,2); 



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);



        j++;  





   if ( j >= db_length+7 ) goto ending;




        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v3 = v_score_load1;


//       v7=_mm_blendv_epi8(v3,v3,mask1);
       v1=_mm_blendv_epi8(v1,v3,mask2);
       v0=_mm_blendv_epi8(v0,v1,mask3);

v_score     = v0;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

   v_score_load1 = _mm_slli_si128(v_score_load2,8); 




            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
  


        p += 16; 



   
            Fup    =  _mm_slli_si128(F,2); 
            Hup1    = _mm_slli_si128(H,2); 



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);



        j++;  




   if ( j >= db_length+7 ) goto ending;




        // prefetch next residue
        k2          = *(p_dbseq-4);
        k          = *p_dbseq++;

        // Create the actual diagonal score vector
        // and update the queue of incomplete score vectors


v0 = v_score_load1;


//       v0=_mm_blendv_epi8(v0,v0,mask1);
       v2=_mm_blendv_epi8(v2,v0,mask2);
       v1=_mm_blendv_epi8(v1,v2,mask3);

v_score     = v1;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));



   v_score_load2 =  _mm_load_si128( (__m128i *) ( query_profile_word + 8*k2 ) ); 

   v_score_load1 = _mm_slli_si128(v_score_load2,8); 




            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
  


        p += 16; 



   
            Fup    =  _mm_slli_si128(F,2); 
            Hup2    = _mm_slli_si128(H,2); 



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup2,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup1,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);



        j++;  




   if ( j >= db_length+7 ) goto ending;





v1 = v_score_load1;


//       v1=_mm_blendv_epi8(v1,v1,mask1);
       v3=_mm_blendv_epi8(v3,v1,mask2);
       v2=_mm_blendv_epi8(v2,v3,mask3);

v_score     = v2;
E   = _mm_max_epu16(_mm_subs_epu16(E,v_gapextend),_mm_subs_epu16(H,v_gapopen));



  

  




            // save old values of F and H to use on next row
      _mm_store_si128( (__m128i *) ( p - 128 ), F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


  
  


        p += 16; 



   
            Fup    =  _mm_slli_si128(F,2); 
            Hup1    = _mm_slli_si128(H,2); 



        F   = _mm_subs_epu16(Fup,v_gapextend);
        tmp = _mm_subs_epu16(Hup1,v_gapopen);
        F   = _mm_max_epu16(F,tmp);
        
        H   =  _mm_adds_epu16(Hup2,v_score);
        H   =  _mm_subs_epu16(H,v_bias);
        
        H   = _mm_max_epu16(H,E);
        H   = _mm_max_epu16(H,F);




 set_matrix_values_from_diagonal_word_vector(E,Ematrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(F,Fmatrix,i,query_length,dbindex,db_length);
 set_matrix_values_from_diagonal_word_vector(H,Hmatrix,i,query_length,dbindex,db_length);
 dbindex++;


        
        // Update highest score encountered this far
        v_maxscore = _mm_max_epu16(v_maxscore,H);



        j++;  






ending:
      _mm_store_si128( (__m128i *) (p - 128) , F);
      _mm_store_si128((__m128i *)( p -128 + 8 ), H);


        query_profile_word += 8*ALPHABET_SIZE;
    }

    // find largest score in the v_maxscore vector

    tmp = _mm_alignr_epi8(v_maxscore,v_maxscore,8);
    v_maxscore = _mm_max_epu16(v_maxscore,tmp);
    tmp = _mm_alignr_epi8(v_maxscore,v_maxscore,4);
    v_maxscore = _mm_max_epu16(v_maxscore,tmp);
    tmp = _mm_alignr_epi8(v_maxscore,v_maxscore,2);
    v_maxscore = _mm_max_epu16(v_maxscore,tmp);


    // store in temporary variable
    score=_mm_extract_epi16(v_maxscore,0);
    
    // return largest score
    return score;
}


