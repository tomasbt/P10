-- vhdl FSM

-- 3 clk system - check par_sche_alloc.odg file 26 states
-- kun en af mean filterne

...
  type states is (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11);
  signal state, next_state : states;

begin
  process (clk)  -- Sequential process
  begin
    if rising_edge(clk) then
      state <= next_state;
    end if;
  end process;

  process (state, reset)  -- Combinational process
  begin
    if reset = '1' then
      next_state <= s1;
    else
      case state is
        when s1 =>
          add1_in1 <= I1;
          add1_in2 <= I_d1;
          add1_add <= '0';
          add1_en <= '1';
          tmp1 <= add1_out;
          ...
          add7_in1 <= I7;
          add7_in2 <= I_d7;
          add7_add <= '0';
          add7_en <= '1';
          tmp7 <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s2;
        when s2 =>
          add1_in1 <= tmp1;
          add1_in2 <= cumsum1;
          add1_add <= '1';
          add1_en <= '1';
          cumsum1 <= add1_out;
          ...
          add6_in1 <= tmp6;
          add6_in2 <= cumsum6;
          add6_add <= '1';
          add6_en <= '1';
          cumsum6 <= add6_out;

          add7_in1 <= I8;
          add7_in2 <= I_d8;
          add7_add <= '0';
          add7_en <= '1';
          tmp8 <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s3;
        when s3 =>
          add1_in1 <= tmp7;
          add1_in2 <= cumsum7;
          add1_add <= '1';
          add1_en <= '1';
          cumsum7 <= add1_out;

          add2_in1 <= tmp8;
          add2_in2 <= cumsum8;
          add2_add <= '1';
          add2_en <= '1';
          cumsum8 <= add2_out;

          add3_in1 <= I9;
          add3_in2 <= I_d9;
          add3_add <= '0';
          add3_en <= '1';
          tmp9 <= add2_out;
          ...
          add7_in1 <= I13;
          add7_in2 <= I_d13;
          add7_add <= '0';
          add7_en <= '1';
          tmp13 <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s4;

        when s4 =>
          add1_in1 <= tmp9;
          add1_in2 <= cumsum9;
          add1_add <= '1';
          add1_en <= '1';
          cumsum9 <= add1_out;
          ...
          add4_in1 <= tmp12;
          add4_in2 <= cumsum12;
          add4_add <= '1';
          add4_en <= '1';
          cumsum12 <= add4_out;

          add5_in1 <= I14;
          add5_in2 <= I_d14;
          add5_add <= '0';
          add5_en <= '1';
          tmp14 <= add5_out;
          ...
          add7_in1 <= I16;
          add7_in2 <= I_d16;
          add7_add <= '0';
          add7_en <= '1';
          tmp16 <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s5;
        when s5 =>
          add1_in1 <= cumsum1;
          add1_in2 <= cumsum2;
          add1_add <= '1';
          add1_en <= '1';
          tmp1_2 <= add1_out;

          add2_in1 <= tmp13;
          add2_in2 <= cumsum13;
          add2_add <= '1';
          add2_en <= '1';
          cumsum13 <= add2_out;
          ...
          add5_in1 <= tmp16;
          add5_in2 <= cumsum16;
          add5_add <= '1';
          add5_en <= '1';
          cumsum16 <= add5_out;

          add6_in1 <= I17;
          add6_in2 <= I_d17;
          add6_add <= '0';
          add6_en <= '1';
          tmp17 <= add6_out;

          add7_in1 <= I18;
          add7_in2 <= I_d18;
          add7_add <= '0';
          add7_en <= '1';
          tmp18 <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s6;
        when s6 =>
          add1_in1 <= cumsum3;
          add1_in2 <= cumsum4;
          add1_add <= '1';
          add1_en <= '1';
          tmp3_4 <= add1_out;
          ...
          add7_in1 <= cumsum15;
          add7_in2 <= cumsum16;
          add7_add <= '1';
          add7_en <= '1';
          tmp15__16 <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s7;
        when s7 =>
          add1_in1 <= tmp1_2;
          add1_in2 <= tmp3_4;
          add1_add <= '1';
          add1_en <= '1';
          tmp1_4 <= add1_out;
          ...
          add4_in1 <= tmp13_14;
          add4_in2 <= tmp15_16;
          add4_add <= '1';
          add4_en <= '1';
          tmp13_16 <= add4_out;

          add5_in1 <= tmp17;
          add5_in2 <= cumsum17;
          add5_add <= '1';
          add5_en <= '1';
          cumsum17 <= add5_out;

          add6_in1 <= tmp18;
          add6_in2 <= cumsum18;
          add6_add <= '1';
          add6_en <= '1';
          cumsum18 <= add6_out;

          add7_in1 <= I19;
          add7_in2 <= I_d19;
          add7_add <= '0';
          add7_en <= '1';
          tmp19 <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s8;

        when s8 =>
          add1_in1 <= tmp1_4;
          add1_in2 <= tmp5_8;
          add1_add <= '1';
          add1_en <= '1';
          tmp1_8 <= add1_out;

          add2_in1 <= tmp9_12;
          add2_in2 <= tmp13_16;
          add2_add <= '1';
          add2_en <= '1';
          tmp9_16 <= add2_out;

          add3_in1 <= cumsum17;
          add3_in2 <= cumsum18;
          add3_add <= '1';
          add3_en <= '1';
          tmp17_18 <= add3_out;

          add4_in1 <= tmp19;
          add4_in2 <= cumsum19;
          add4_add <= '1';
          add4_en <= '1';
          cumsum19 <= add4_out;

          add5_in1 <= X;
          add5_in2 <= X;
          add5_add <= X;
          add5_en <= '0';
          X <= add5_out;
          ...
          add7_in1 <= X;
          add7_in2 <= X;
          add7_add <= X;
          add7_en <= '0';
          X <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s9;
        when s9 =>
          add1_in1 <= tmp1_8;
          add1_in2 <= tmp9_16;
          add1_add <= '1';
          add1_en <= '1';
          tmp1_16 <= add1_out;

          add2_in1 <= tmp17_18;
          add2_in2 <= tmp19;
          add2_add <= '1';
          add2_en <= '1';
          tmp17_19 <= add2_out;

          add3_in1 <= X;
          add3_in2 <= X;
          add3_add <= X;
          add3_en <= '0';
          X <= add3_out;
          ...
          add7_in1 <= X;
          add7_in2 <= X;
          add7_add <= X;
          add7_en <= '0';
          X <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s10;
        when s10 =>
          add1_in1 <= tmp1_16;
          add1_in2 <= tmp17_19;
          add1_add <= '1';
          add1_en <= '1';
          tmp1_19 <= add1_out;

          add2_in1 <= X;
          add2_in2 <= X;
          add2_add <= X;
          add2_en <= '0';
          X <= add2_out;
          ...
          add7_in1 <= X;
          add7_in2 <= X;
          add7_add <= X;
          add7_en <= '0';
          X <= add7_out;

          mult_in1 <= X;
          mult_in2 <= X;
          mult_en <= '0';
          X <= mult_out;
          next_state <= s11;
        when s11 =>
          add1_in1 <= X;
          add1_in2 <= X;
          add1_add <= X;
          add1_en <= '0';
          X <= add1_out;
          ...
          add7_in1 <= X;
          add7_in2 <= X;
          add7_add <= X;
          add7_en <= '0';
          X <= add7_out;

          mult_in1 <= tmp1_19;
          mult_in2 <= 0.00277;
          mult_en <= '1';
          Out1 <= mult_out;
          next_state <= s1;
      end case;
    end if;
  end process;

  add1_add
  add1_en
  add1_in1
  add1_in2
  add1_out
