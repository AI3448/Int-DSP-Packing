#ifndef __CONV2D_DSPOPT_HPP__
#define __CONV2D_DSPOPT_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "debug.hpp"
#include "function.h"
#include "stream_tools.h"

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void stream_in_row(
    stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    ap_uint<2> rowBufferIdx) {
// #pragma HLS inline off
  // ap_uint<IN_PE *IN_BIT> reg = 0;

  for (ap_uint<7> peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++)
    for (ap_uint<9> w = 0; w < IN_W / 2; w++) {
#pragma HLS pipeline
      ap_uint<IN_PE * IN_BIT * 2> data;
      ap_uint<IN_PE * IN_BIT> data0, data1;
      data = in.read();
      row_buffer[peIdx % (SIMD / IN_PE)][rowBufferIdx]
                [w * IN_CH / SIMD + peIdx / (SIMD / IN_PE)] = data;
    }
}

template <unsigned ROW, unsigned COL, unsigned CH, 
          unsigned BIT, unsigned PE>
void splitData(stream<ap_uint<PE * BIT * 2>> &in,
                     stream<ap_uint<PE * BIT>> &out,
                     const unsigned rep = 1) {
// #pragma HLS inline
  const unsigned loop_r = ROW * 1;

  for (unsigned r = 0; r < ROW * rep; r++)
#pragma HLS LOOP_TRIPCOUNT min=loop_r max=loop_r
    for (unsigned peIdx = 0; peIdx < CH / PE; peIdx++)
      for (unsigned w = 0; w < COL; w += 2) {

#pragma HLS pipeline II = 2
        ap_uint<BIT * PE> data0, data1;
        (data1, data0) = in.read();

        out.write(data0);
        out.write(data1);
      }
}

template <unsigned ROW, unsigned COL, unsigned CH, 
          unsigned BIT, unsigned PE>
void packData(stream<ap_uint<PE * BIT>> &in,
                     stream<ap_uint<PE * BIT * 2>> &out,
                     const unsigned rep = 1) {
// #pragma HLS inline

  const unsigned loop_r = ROW * 1;

  for (unsigned r = 0; r < ROW * rep; r++)
#pragma HLS LOOP_TRIPCOUNT min=loop_r max=loop_r
    for (unsigned peIdx = 0; peIdx < CH / PE; peIdx++)
      for (unsigned w = 0; w < COL; w += 2) {

#pragma HLS pipeline II = 2
        ap_uint<BIT * PE> data0, data1;
        data0 = in.read();
        data1 = in.read();

        out.write((data1, data0));
      }
}


template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void stream_in_row_lut(
    stream<ap_uint<IN_PE * IN_BIT>> &in,
    ap_uint<IN_PE * IN_BIT> row_buffer[SIMD / IN_PE][4]
                                          [(IN_W + 1) * IN_CH / SIMD],
    ap_uint<2> rowBufferIdx) {
// #pragma HLS inline off
  // ap_uint<IN_PE *IN_BIT> reg = 0;

  for (unsigned peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++)
    for (unsigned w = 0; w < IN_W + 1; w++) {
#pragma HLS pipeline
      ap_uint<IN_PE * IN_BIT> data;
      if (w != IN_W) {
        data = in.read();
      } else {
        data = 0;
      }
      row_buffer[peIdx % (SIMD / IN_PE)][rowBufferIdx]
                [w * IN_CH / SIMD + peIdx / (SIMD / IN_PE)] = data;
    }
}


template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned PE>
void shiftOut(
    stream<ap_uint<PE * IN_BIT * 2>> &in,
    stream<ap_uint<PE * IN_BIT * 2>> &out,
    const unsigned reps = 1) {

  const unsigned int loop_h = IN_H * 1;
  ap_uint<PE * IN_BIT> reg = 0;
  ap_uint<PE * IN_BIT * 2> data;
  ap_uint<PE * IN_BIT> data0, data1;
  (data1, data0) = in.read();
  reg = data1;

  for (unsigned int h = 0; h < IN_H * reps; h++) {
#pragma HLS LOOP_TRIPCOUNT min=loop_h max=loop_h
    for (unsigned peIdx = 0; peIdx < IN_CH / PE; peIdx++)
      for (unsigned w = 0; w < IN_W / 2; w++) {
  #pragma HLS pipeline
        (data1, data0) = in.read();
        data = (data0, reg);
        reg = data1;
        out.write(data);
      }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void stream_out_data(
    stream<ap_uint<SIMD * IN_BIT * 2>> &out,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned WLEN = IN_W / 2;

  ap_uint<4> infoldIdx = 0;
  ap_uint<8> w = 0;

  for (ap_uint<7> peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (ap_uint<12> cycle = 0; cycle < WLEN * K * SIMDNUM; cycle++) {
      // for (unsigned w = 0; w < WLEN; w++) {
      //   for (unsigned wr = 0; wr < K; wr++) {
      //     for (unsigned simdIdx = 0; simdIdx < SIMDNUM; simdIdx++) {
      ap_uint<2> wr = infoldIdx / SIMDNUM;
      ap_uint<6> simdIdx = infoldIdx % SIMDNUM;
#pragma HLS pipeline
      ap_uint<SIMD * IN_BIT> data0;
      ap_uint<SIMD * IN_BIT> data1;
      ap_uint<IN_PE * IN_BIT * 2> buffer_data[SIMD / IN_PE];
#pragma HLS array_partition variable = buffer_data complete
      ap_uint<2> rowBufferIdx = startRowBufferIdx + wr;
      for (int i = 0; i < SIMD / IN_PE; i++) {
#pragma HLS unroll
        buffer_data[i] = row_buffer[i][rowBufferIdx][w * SIMDNUM + simdIdx];
      }

      if (outRowIdx + wr == 0 || outRowIdx + wr == IN_H + 1) {
        data0 = 0;
        data1 = 0;
      } else {
        for (int i = 0; i < SIMD / IN_PE; i++) {
          data0((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
              buffer_data[i](IN_PE_BIT - 1, 0);
          data1((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
              buffer_data[i](IN_PE_BIT * 2 - 1, IN_PE_BIT);
        }
      }
      out.write((data1, data0));

      if (cycle == WLEN * K * SIMDNUM - 1) {
        w = 0;
      } else if (infoldIdx == K * SIMDNUM - 1) {
        w++;
      }

      if (infoldIdx == K * SIMDNUM - 1) {
        infoldIdx = 0;
      } else {
        infoldIdx++;
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void stream_out_data_lut(
    stream<ap_uint<SIMD * IN_BIT * K>> &out,
    ap_uint<IN_PE * IN_BIT> row_buffer[SIMD / IN_PE][4]
                                          [(IN_W + 1) * IN_CH / SIMD],
    ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete
#pragma HLS array_partition variable = row_buffer dim = 2 complete


  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned WLEN = IN_W / 2;

  ap_uint<8> infoldIdx = 0;
  ap_uint<8> w = 0;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (unsigned cycle = 0; cycle < WLEN * K * SIMDNUM; cycle++) {
      // for (unsigned w = 0; w < WLEN; w++) {
      //   for (unsigned wr = 0; wr < K; wr++) {
      //     for (unsigned simdIdx = 0; simdIdx < SIMDNUM; simdIdx++) {
      ap_uint<2> wr = infoldIdx / SIMDNUM;
      ap_uint<4> simdIdx = infoldIdx % SIMDNUM;
#pragma HLS pipeline
      ap_uint<SIMD * IN_BIT> data0;
      ap_uint<SIMD * IN_BIT> data1;
      ap_uint<SIMD * IN_BIT> data2;

      ap_uint<IN_PE * IN_BIT> buffer_data[3][SIMD / IN_PE];
#pragma HLS array_partition variable = buffer_data complete dim = 1
#pragma HLS array_partition variable = buffer_data complete dim = 2
      
      for (unsigned i = 0; i < K; i++) {
        for (unsigned j = 0; j < SIMD / IN_PE; j++) {
#pragma HLS unroll
          buffer_data[i][j] = row_buffer[j][startRowBufferIdx + i][(w + wr) * SIMDNUM + simdIdx];
        }
      }

      if (outRowIdx == 0)
        data0 = 0;
      else
        for (unsigned i = 0; i < SIMD / IN_PE; i++) {
          data0((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) = buffer_data[0][i];
        }
      for (unsigned i = 0; i < SIMD / IN_PE; i++) {
          data1((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) = buffer_data[1][i];
        }
      if (outRowIdx == IN_H - 1)
        data2 = 0;
      else
        for (unsigned i = 0; i < SIMD / IN_PE; i++) {
          data2((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) = buffer_data[2][i];
        }


      out.write((data2, data1, data0));

      if (cycle == WLEN * K * SIMDNUM - 1) {
        w = 0;
      } else if (infoldIdx == K * SIMDNUM - 1) {
        w += 2;
      }

      if (infoldIdx == K * SIMDNUM - 1) {
        infoldIdx = 0;
      } else {
        infoldIdx++;
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv3padding(stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
                  stream<ap_uint<SIMD * IN_BIT * 2>> &out,
                  const unsigned reps = 1) {
  static_assert(SIMD % IN_PE == 0, "SIMD %IN_PE !=0");
  static_assert(K == 3, "K!=3");

  ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                        [IN_W / 2 * IN_CH / SIMD];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM
  ap_uint<8> inh = 0;
  ap_uint<8> outh = 0;

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 3;
  ap_int<10> rowIdx = 0;

  const unsigned loop_rep = 1 * IN_H - 2;

  stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, storeBufferIdx);
  storeBufferIdx++;
  
  stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, storeBufferIdx);
  storeBufferIdx++;

  for (unsigned rep = 0; rep < reps * IN_H - 2; rep++) {
#pragma HLS LOOP_TRIPCOUNT min=loop_rep max=loop_rep
#pragma HLS dependence intra false variable = row_buffer
    stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, storeBufferIdx);
    stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, rowIdx, loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
  stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
  
  stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv3padding_lut(stream<ap_uint<IN_PE * IN_BIT>> &in,
                  stream<ap_uint<SIMD * IN_BIT * K>> &out,
                  const unsigned reps = 1) {
  static_assert(SIMD % IN_PE == 0, "SIMD %IN_PE !=0");
  static_assert(K == 3, "K!=3");

  ap_uint<IN_PE * IN_BIT> row_buffer[SIMD / IN_PE][4]
                                        [(IN_W + 1) * IN_CH / SIMD];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM
  ap_uint<8> inh = 0;
  ap_uint<8> outh = 0;

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 3;
  ap_int<10> rowIdx = 0;

  const unsigned loop_rep = 1 * IN_H - 2;

  stream_in_row_lut<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, storeBufferIdx);
  storeBufferIdx++;
  
  stream_in_row_lut<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, storeBufferIdx);
  storeBufferIdx++;

  for (unsigned rep = 0; rep < reps * IN_H - 2; rep++) {
#pragma HLS LOOP_TRIPCOUNT min=loop_rep max=loop_rep
#pragma HLS dependence intra false variable = row_buffer
    stream_in_row_lut<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, storeBufferIdx);
    stream_out_data_lut<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, rowIdx, loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
  stream_out_data_lut<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
  
  stream_out_data_lut<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
}


template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned INFOLD, unsigned PE>
void streamBnRelu(stream<ap_uint<PE * M_BIT * 2>> &in,
                  const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                  const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                  stream<ap_uint<PE * OUT_BIT * 2>> &out,
                  const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
  ap_uint<PE * M_BIT> reg;
  ap_uint<PE * M_BIT * 2> data_in;
  ap_uint<PE * M_BIT> data0, data1;
  ap_uint<PE * OUT_BIT * 2> data_out;
  (data1, data0) = in.read();
  reg = data1;

  const unsigned loop_r = 1 * OUT_ROW;

  for (int r = 0; r < OUT_ROW * rep; r++)
#pragma HLS LOOP_TRIPCOUNT min=loop_r max=loop_r
    for (ap_uint<7> peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (ap_uint<9> w = 0; w < OUT_COL / 2; w++) {
#pragma HLS pipeline II = INFOLD
        ap_int<M_BIT> invec[2 * PE];
#pragma HLS array_partition variable = invec dim = 1 complete
        (data1, data0) = in.read();
        data_in = (data0, reg);
        reg = data1;
        for (int i = 0; i < PE * 2; i++) {
          invec[i] = data_in((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE * 2; i++) {
          data_out((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i % PE][peIdx],
                                       bias[i % PE][peIdx]);
        }
        out.write(data_out);
      }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned INFOLD, unsigned PE>
void streamBnRelu_l4567(stream<ap_uint<PE * M_BIT * 2>> &in,
                  const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                  const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                  stream<ap_uint<PE * OUT_BIT * 2>> &out,
                  const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
  ap_uint<PE * M_BIT> reg;
  ap_uint<PE * M_BIT * 2> data_in;
  ap_uint<PE * M_BIT> data0, data1;
  ap_uint<PE * OUT_BIT * 2> data_out;
  (data1, data0) = in.read();
  reg = data1;

  const unsigned loop_r = 1 * OUT_ROW;
  const unsigned PFOLD = INFOLD / PE / 2;


  for (int r = 0; r < OUT_ROW * rep; r++)
#pragma HLS LOOP_TRIPCOUNT min=loop_r max=loop_r
    for (int peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (int w = 0; w < OUT_COL / 2; w++) {
        for (int i = 0; i < PE * 2; i++) {
#pragma HLS pipeline II = PFOLD
          ap_int<M_BIT> invec[2 * PE];
#pragma HLS array_partition variable = invec dim = 1 complete
          if(i == 0) {
            (data1, data0) = in.read();
            data_in = (data0, reg);
            reg = data1;
          }
          invec[i] = data_in((i + 1) * M_BIT - 1, i * M_BIT);
          data_out((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i % PE][peIdx],
                                        bias[i % PE][peIdx]);
          if(i == PE * 2 - 1) {
            out.write(data_out);
          }
        }
      }
}



template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_input_data_int(ap_uint<IN_BIT * SIMD> A, ap_uint<IN_BIT * SIMD> B,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD]) {
#pragma HLS array_partition variable = ipack

  for (int i = 0; i < SIMD; i++) {
    ipack[i] =(A(i * IN_BIT + IN_BIT - 1, i * IN_BIT), (ap_uint<PROD_BIT - IN_BIT>)0,B(i * IN_BIT + IN_BIT - 1, i * IN_BIT));
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_weight_data(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0,
                      ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS array_partition variable = wpack

  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] =
        (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_weight_data_int(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0,
                      ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS array_partition variable = wpack

  for (int i = 0; i < SIMD; i++) {
    ap_uint<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_uint<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_uint<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_uint<1> s2 =  w2[i * W_BIT + W_BIT - 1];
    ap_uint<W_BIT> w1_seg_tmp = w1_seg - s2;
    ap_uint<1> s1 =  w1_seg_tmp[W_BIT-1];
    ap_uint<W_BIT> w0_seg_tmp = w0_seg - s1;
//    ap_uint<PROD_BIT - W_BIT> m1 = ((ap_uint<PROD_BIT - W_BIT>)s1);
//    ap_uint<PROD_BIT - W_BIT>     m2 = (ap_uint<PROD_BIT - W_BIT>)s2;
    ap_uint<PROD_BIT - W_BIT> m1 = (s1,s1,s1,s1,s1,s1,s1);
    ap_uint<PROD_BIT - W_BIT>     m2 = (s2,s2,s2,s2,s2,s2,s2);
    wpack[i] = (w0_seg_tmp, m1, w1_seg_tmp, m2, w2_seg);
  }
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned CASCADE>
void simd_MAC_int(ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD],
              ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
              ap_int<PROD_BIT + 5> &partial0, ap_int<PROD_BIT + 5> &partial1,
              ap_int<PROD_BIT + 5> &partial2, ap_int<PROD_BIT + 5> &partial3) {
#pragma HLS ARRAY_PARTITION variable = wpack complete
#pragma HLS ARRAY_PARTITION variable = ipack complete
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  ap_uint<5> cnt0 = 0;
  ap_uint<5> cnt1 = 0;
  ap_uint<5> cnt2 = 0;
  for (int i = 0; i < SIMD; i += CASCADE) {
//#pragma HLS unroll
    ap_int<PROD_BIT * 4> m = 0;
    for (int cs = 0; cs < CASCADE; cs++) {
      m += wpack[i + cs] * ipack[i + cs];
    }

    ap_int<PROD_BIT> tmp0 = m(PROD_BIT - 1, 0);
    ap_int<PROD_BIT> tmp1 = m(PROD_BIT * 2 - 1, PROD_BIT);
    ap_int<PROD_BIT> tmp2 = m(PROD_BIT * 3 - 1, PROD_BIT * 2);
    ap_int<PROD_BIT> tmp3 = m(PROD_BIT * 4 - 1, PROD_BIT * 3);

    ap_int<PROD_BIT> p0 = tmp0;
    ap_int<PROD_BIT> p1 = tmp1;
    ap_int<PROD_BIT> p2 = tmp2;
    ap_int<PROD_BIT> p3 = tmp3;

    cnt0 += tmp0[PROD_BIT - 1];
    cnt1 += tmp1[PROD_BIT - 1];
    cnt2 += tmp2[PROD_BIT - 1];

    r0 += p0;
    r1 += p1;
    r2 += p2;
    r3 += p3;
  }
  partial0 = r0;
  partial1 = r1 + cnt0;
  partial2 = r2 + cnt1;
  partial3 = r3 + cnt2;
}

template <unsigned IN_BIT, unsigned W_BIT>
ap_int<IN_BIT + W_BIT> conv_mul_lut(ap_uint<IN_BIT> in, ap_int<W_BIT> w) {
  ap_int<IN_BIT + W_BIT> out;
#pragma HLS RESOURCE variable=return core=Mul_LUT
#pragma HLS inline off
  out = in * w;
  return out;
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void simd_MAC_DSPLUT(ap_int<W_BIT * SIMD> w0, ap_int<W_BIT * SIMD> w1,
                     ap_int<W_BIT * SIMD> w2, ap_uint<IN_BIT * SIMD> i0,
                     ap_uint<IN_BIT * SIMD> i1, ap_int<PROD_BIT + 5> &partial0,
                     ap_int<PROD_BIT + 5> &partial1,
                     ap_int<PROD_BIT + 5> &partial2,
                     ap_int<PROD_BIT + 5> &partial3) {
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w0_seg = w0((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w2_seg = w2((i + 1) * W_BIT - 1, i * W_BIT);
    ap_uint<IN_BIT> x0_seg = i0((i + 1) * IN_BIT - 1, i * IN_BIT);
    ap_uint<IN_BIT> x1_seg = i1((i + 1) * IN_BIT - 1, i * IN_BIT);

    r0 += conv_mul_lut<IN_BIT, W_BIT>(x0_seg, w2_seg);
    r1 += conv_mul_lut<IN_BIT, W_BIT>(x0_seg, w1_seg) + conv_mul_lut<IN_BIT, W_BIT>(x1_seg, w2_seg);
    r2 += conv_mul_lut<IN_BIT, W_BIT>(x0_seg, w0_seg) + conv_mul_lut<IN_BIT, W_BIT>(x1_seg, w1_seg);
    r3 += conv_mul_lut<IN_BIT, W_BIT>(x1_seg, w0_seg);
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

template <unsigned K, unsigned IN_BIT, unsigned IN_CH, unsigned OUT_BIT,
          unsigned OUT_H, unsigned OUT_W, unsigned OUT_CH, unsigned W_BIT,
          unsigned GUARD_BIT, unsigned M_BIT, unsigned INC_BIT, // GUARD_BIT恒等于3，所以cascade不能超过4，因为考虑是int计算时，两个4bit负数需要用9bit数来存储如-8*-8=64,8bit数只能到63，所以当int计算时需要保护位需要+1。当换成uint计算时则不需要
          unsigned BIAS_BIT, unsigned SIMD, unsigned CASCADE, unsigned PE,
          unsigned L_SHIFT>
void convDSPOpt_int(
    stream<ap_uint<SIMD * IN_BIT * 2>> &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][3][K * IN_CH / SIMD * OUT_CH / PE],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    // stream<ap_uint<PE * OUT_BIT * 2>> &out,
    stream<ap_uint<PE * M_BIT * 2>> &out,
    const unsigned reps = 1) {

  static_assert(IN_CH % SIMD == 0, "IN_CH % SIMD !=0");
  static_assert(SIMD % CASCADE == 0, "SIMD % CASCADE != 0");
  static_assert(CASCADE <= 4, "SIMD % CASCADE != 0");
  const unsigned PENUM = OUT_CH / PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = W_BIT * 3 + IN_BIT * 2 + GUARD_BIT * 2;
  const unsigned IPACK_BIT = IN_BIT * 2 + W_BIT + GUARD_BIT * 1;
  const unsigned OUT_WNUM = OUT_W / 2;
  const unsigned INFOLD = K * SIMDNUM;

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  ap_int<WPACK_BIT> wpacks[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  ap_uint<IPACK_BIT> ipack[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipack complete dim = 1

  // ap_uint<12> weightAddr = 0;
  ap_int<M_BIT> firPartialRes0[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes0 complete dim = 1
  ap_int<M_BIT> firPartialRes1[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes1 complete dim = 1

  ap_int<M_BIT> outPartialArr0[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr0 complete dim = 1
  ap_int<M_BIT> outPartialArr1[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr1 complete dim = 1

  const unsigned int loop_h = OUT_H * 1;

  ap_int<PROD_BIT + 5> firPartial0;
  ap_int<PROD_BIT + 5> firPartial1;
  ap_int<PROD_BIT + 5> firPartial2;
  ap_int<PROD_BIT + 5> firPartial3;
  for (unsigned int h = 0; h < OUT_H * reps; h++) {
#pragma HLS LOOP_TRIPCOUNT min=loop_h max=loop_h
    for (ap_uint<7> peIdx = 0; peIdx < PENUM; peIdx++) {
      for (ap_uint<9> w = 0; w < OUT_WNUM; w++) {
        for (ap_uint<5> infoldIdx = 0; infoldIdx < INFOLD; infoldIdx++) {
#pragma HLS pipeline
          bool m_clear = (w == 0);
          bool o_clear = (infoldIdx == 0);
          bool o_out = (infoldIdx == INFOLD - 1);
          ap_uint<SIMD * IN_BIT> data1, data0;
          (data1, data0) = vec.read();
          pack_input_data_int<IN_BIT, SIMD, PROD_BIT>(data1, data0, ipack);
          for (int p = 0; p < PE; p++) {
#pragma HLS unroll
            pack_weight_data_int<W_BIT, SIMD, PROD_BIT>(
                weights[p][2][peIdx * INFOLD + infoldIdx],
                weights[p][1][peIdx * INFOLD + infoldIdx],
                weights[p][0][peIdx * INFOLD + infoldIdx], wpacks[p]);
          }

          for (int p = 0; p < PE; p++) {
            // cout << "FIR result compare " << endl;
#pragma HLS unroll
            simd_MAC_int<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE>(
                wpacks[p], ipack, firPartial0, firPartial1, firPartial2,
                firPartial3);
            // getchar();
            if (m_clear & o_clear) { 
              outPartialArr0[p] = firPartialRes0[p];
              outPartialArr1[p] = firPartial1;
            }
            if (m_clear & !o_clear) {
              outPartialArr0[p] = outPartialArr0[p];
              outPartialArr1[p] += firPartial1;
            }
            if (!m_clear & o_clear) { 
              outPartialArr0[p] = firPartial0 + firPartialRes0[p];
              outPartialArr1[p] = firPartial1 + firPartialRes1[p];
            }
            if (!m_clear & !o_clear) {
              outPartialArr0[p] += firPartial0;
              outPartialArr1[p] += firPartial1;
            }

            if (o_clear) {
              firPartialRes0[p] = firPartial2;
              firPartialRes1[p] = firPartial3;
            }
            else {
              firPartialRes0[p] += firPartial2;
              firPartialRes1[p] += firPartial3;
            }


          }

          if (o_out) {
            ap_uint<PE * M_BIT> out_buf0;
            ap_uint<PE * M_BIT> out_buf1;
            for (int p = 0; p < PE; p++) {
#pragma HLS unroll
              out_buf0(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr0[p];
              out_buf1(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr1[p];

            }
            // out.write((oData1, oData0));
            out.write((out_buf1, out_buf0));
          }
        }
      }
    }
  }
  ap_uint<PE * M_BIT> out_buf2;
  for (ap_uint<4> p = 0; p < PE; p++) {
#pragma HLS unroll
    out_buf2(p * M_BIT + M_BIT - 1, p * M_BIT) = firPartialRes0[p];
  }
  out.write((0, out_buf2));
}

template <unsigned K, unsigned IN_BIT, unsigned IN_CH, unsigned OUT_BIT,
          unsigned OUT_H, unsigned OUT_W, unsigned OUT_CH, unsigned W_BIT,
          unsigned GUARD_BIT, unsigned M_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT, unsigned SIMD, unsigned CASCADE, unsigned PE,
          unsigned L_SHIFT>
void convDSPLUTOpt(
    stream<ap_uint<SIMD * IN_BIT * 2>> &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][3][K * IN_CH / SIMD * OUT_CH / PE],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    // stream<ap_uint<PE * OUT_BIT * 2>> &out,
    stream<ap_uint<PE * M_BIT * 2>> &out,
    const unsigned reps = 1) {

  static_assert(IN_CH % SIMD == 0, "IN_CH % SIMD !=0");
  static_assert(SIMD % CASCADE == 0, "SIMD % CASCADE != 0");
  static_assert(CASCADE <= 4, "SIMD % CASCADE != 0");
  const unsigned PENUM = OUT_CH / PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = W_BIT * 3 + IN_BIT * 2 + GUARD_BIT * 2;
  const unsigned IPACK_BIT = IN_BIT * 2 + W_BIT + GUARD_BIT * 1;
  const unsigned OUT_WNUM = OUT_W / 2;
  const unsigned INFOLD = K * SIMDNUM;

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1


  ap_int<W_BIT> wunpacks[PE][3][SIMD];
#pragma HLS ARRAY_PARTITION variable = wunpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wunpacks complete dim = 2
#pragma HLS ARRAY_PARTITION variable = wunpacks complete dim = 3

  ap_uint<IN_BIT> iunpack0[SIMD];
#pragma HLS ARRAY_PARTITION variable = iunpack0 complete dim = 1
  ap_uint<IN_BIT> iunpack1[SIMD];
#pragma HLS ARRAY_PARTITION variable = iunpack1 complete dim = 1

  // ap_uint<12> weightAddr = 0;
  ap_int<M_BIT> firPartialRes0[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes0 complete dim = 1
  ap_int<M_BIT> firPartialRes1[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes1 complete dim = 1

  ap_int<M_BIT> outPartialArr0[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr0 complete dim = 1
  ap_int<M_BIT> outPartialArr1[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr1 complete dim = 1

  const unsigned int loop_h = OUT_H * 1;

  ap_int<PROD_BIT + 5> firPartial0;
  ap_int<PROD_BIT + 5> firPartial1;
  ap_int<PROD_BIT + 5> firPartial2;
  ap_int<PROD_BIT + 5> firPartial3;
  for (unsigned int h = 0; h < OUT_H * reps; h++) {
#pragma HLS LOOP_TRIPCOUNT min=loop_h max=loop_h
    for (ap_uint<7> peIdx = 0; peIdx < PENUM; peIdx++) {
      for (ap_uint<9> w = 0; w < OUT_WNUM; w++) {
        for (ap_uint<5> infoldIdx = 0; infoldIdx < INFOLD; infoldIdx++) {
#pragma HLS pipeline
          bool m_clear = (w == 0);
          bool o_clear = (infoldIdx == 0);
          bool o_out = (infoldIdx == INFOLD - 1);
          ap_uint<SIMD * IN_BIT> data1, data0;
          (data1, data0) = vec.read();
          for (int p = 0; p < PE; p++) {
#pragma HLS unroll 
            simd_MAC_DSPLUT<W_BIT, IN_BIT, SIMD, PROD_BIT>(
                weights[p][0][peIdx * INFOLD + infoldIdx],
                weights[p][1][peIdx * INFOLD + infoldIdx],
                weights[p][2][peIdx * INFOLD + infoldIdx], data0, data1,
                firPartial0, firPartial1, firPartial2, firPartial3);
            // getchar();
            if (m_clear & o_clear) {
              outPartialArr0[p] = firPartialRes0[p];
              outPartialArr1[p] = firPartial1;
            }
            if (m_clear & !o_clear) {
              outPartialArr0[p] = outPartialArr0[p];
              outPartialArr1[p] += firPartial1;
            } 
            if (!m_clear & o_clear) {
              outPartialArr0[p] = firPartial0 + firPartialRes0[p];
              outPartialArr1[p] = firPartial1 + firPartialRes1[p];
            }
            if (!m_clear & !o_clear) {
              outPartialArr0[p] += firPartial0;
              outPartialArr1[p] += firPartial1;
            }
            if (o_clear) {
              firPartialRes0[p] = firPartial2;
              firPartialRes1[p] = firPartial3;
            }
            else {
              firPartialRes0[p] += firPartial2;
              firPartialRes1[p] += firPartial3;
            }
          }

          if (o_out) {
            ap_uint<PE * M_BIT> out_buf0;
            ap_uint<PE * M_BIT> out_buf1;
            for (int p = 0; p < PE; p++) {
#pragma HLS unroll 
              out_buf0(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr0[p];
              out_buf1(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr1[p];

            }
            out.write((out_buf1, out_buf0));
          }
        }
      }
    }
  }  
  ap_uint<PE * M_BIT> out_buf2;
  for (ap_uint<4> p = 0; p < PE; p++) {
#pragma HLS unroll
    out_buf2(p * M_BIT + M_BIT - 1, p * M_BIT) = firPartialRes0[p];
  }
  out.write((0, out_buf2));
}

template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,
          unsigned OUT_CH,
          unsigned OUT_BIT,
          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned SIMD, unsigned CASCADE, unsigned IN_PE, unsigned PE,
          unsigned L_SHIFT>
void conv3x3_bn_act_DSPopt_int(
    stream<ap_uint<IN_BIT * IN_PE * 2>> &in,
    const ap_uint<SIMD * W_BIT> weights[PE][3]
                                       [((IN_CH * 3) / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<OUT_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;
  const unsigned INFOLD = 3 * IN_CH / SIMD;

  stream<ap_uint<SIMD * IN_BIT * 2>> padding_out("padding_out");
  conv3padding<3, IN_ROW, IN_COL, IN_CH, IN_BIT, IN_PE, SIMD, OUT_CH / PE>(
      in, padding_out, reps);
  stream<ap_uint<PE * OUT_BIT * 2>> mvau_out("mvau_out");
  stream<ap_uint<PE * M_BIT * 2>> conv_out("conv_out");
  convDSPOpt_int<3, IN_BIT, IN_CH, OUT_BIT, OUT_ROW, OUT_COL, OUT_CH, W_BIT, 3,
             M_BIT, INC_BIT, BIAS_BIT, SIMD, CASCADE, PE, L_SHIFT>(
      padding_out, weights, inc, bias, conv_out, reps);
  streamBnRelu<OUT_ROW, OUT_COL, OUT_CH, M_BIT, OUT_BIT, INC_BIT, BIAS_BIT,
        L_SHIFT, IN_BIT, W_BIT, INFOLD, PE>(conv_out, inc, bias, out,
                                    reps);
}

template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,
          unsigned OUT_CH,
          unsigned OUT_BIT,
          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned SIMD, unsigned CASCADE, unsigned IN_PE, unsigned PE,
          unsigned L_SHIFT>
void conv3x3_bn_act_DSPopt_l4567_int(
    stream<ap_uint<IN_BIT * IN_PE * 2>> &in,
    const ap_uint<SIMD * W_BIT> weights[PE][3]
                                       [((IN_CH * 3) / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<OUT_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;
  const unsigned INFOLD = 3 * IN_CH / SIMD;


  stream<ap_uint<SIMD * IN_BIT * 2>> padding_out("padding_out");
  conv3padding<3, IN_ROW, IN_COL, IN_CH, IN_BIT, IN_PE, SIMD, OUT_CH / PE>(
      in, padding_out, reps);

  stream<ap_uint<PE * OUT_BIT * 2>> mvau_out("mvau_out");
  stream<ap_uint<PE * M_BIT * 2>> conv_out("conv_out");
  convDSPOpt_int<3, IN_BIT, IN_CH, OUT_BIT, OUT_ROW, OUT_COL, OUT_CH, W_BIT, 3,
             M_BIT, INC_BIT, BIAS_BIT, SIMD, CASCADE, PE, L_SHIFT>(
      padding_out, weights, inc, bias, conv_out, reps);


  streamBnRelu_l4567<OUT_ROW, OUT_COL, OUT_CH, M_BIT, OUT_BIT, INC_BIT, BIAS_BIT,
        L_SHIFT, IN_BIT, W_BIT, INFOLD, PE>(conv_out, inc, bias, out,
                                    reps);
}

template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT,

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned CASCADE, unsigned IN_PE, unsigned PE,
          unsigned L_SHIFT>
void conv3x3_bn_act_DSPLUTopt(
    stream<ap_uint<IN_BIT * IN_PE * 2>> &in,
    const ap_uint<SIMD * W_BIT> weights[PE][3]
                                       [((IN_CH * 3) / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<OUT_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;
  const unsigned INFOLD = 3 * IN_CH / SIMD;
  stream<ap_uint<SIMD * IN_BIT * 2>> padding_out("padding_out");
  conv3padding<3, IN_ROW, IN_COL, IN_CH, IN_BIT, IN_PE, SIMD, OUT_CH / PE>(
      in, padding_out, reps);
  stream<ap_uint<PE * OUT_BIT * 2>> mvau_out("mvau_out");
  stream<ap_uint<PE * M_BIT * 2>> conv_out("conv_out");
  convDSPLUTOpt<3, IN_BIT, IN_CH, OUT_BIT, OUT_ROW, OUT_COL, OUT_CH, W_BIT, 3,
             M_BIT, INC_BIT, BIAS_BIT, SIMD, CASCADE, PE, L_SHIFT>(
      padding_out, weights, inc, bias, conv_out, reps);
  streamBnRelu<OUT_ROW, OUT_COL, OUT_CH, M_BIT, OUT_BIT, INC_BIT, BIAS_BIT,
      L_SHIFT, IN_BIT, W_BIT, INFOLD, PE>(conv_out, inc, bias, out,
                                  reps);
}


#endif
