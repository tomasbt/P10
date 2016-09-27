-- vhdl test code for fsm

library ieee;               -- add this ti the IEEE library
use ieee.std_logic_1164.all -- includes std_logic

entity b_block is
  generic(n1: natural := 8);
  port(
    clk, reset      : in std_logic;
    mu_p, a_o, mu_I : in std_logic_vector(n-1 downto 0);
    b_o             : out std_logic_vector(n-1 downto 0)
  );

architecture behaviour of b_block is
  signal mult_out : std_logic_vector(n-1 downto 0);

  type state_type is (s1, s2);
  signal state, next_state : state_type;

begin

  state_reg: process (clk)
  begin
    if rising_edge(clk) then
      state <= next_state;
    end if;
  end process;

  comb_logic: process (state, reset)
  begin
    if reset = '1' then
      next_state <= s1;
    else
      case state is
        when s1 =>
          mult_out <= mu_I * a_o;
          next_state <= s2;
        when s2 =>
          b_o <= mult_out + mu_p;
          next_state <= s1;
      end case;
    end if;
  end process;

  
end behaviour;
