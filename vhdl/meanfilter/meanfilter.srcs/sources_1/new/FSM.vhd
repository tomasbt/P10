----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 05/17/2016 04:02:07 PM
-- Design Name: 
-- Module Name: FSM - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity FSM is
--  Port ( );
    generic(
        x_min : integer := 0;
        x_max : integer := 749;
        y_min : integer := 0;
        y_max : integer := 497
        );
    port(
        clk, reset : in std_logic
        );
end FSM;

architecture Behavioral of FSM is
    type state_type is (idle,senddata,decision); -- Collection of states
    signal curr_state, next_state : state_type; 
    signal ready : std_logic;
    signal x_cor, y_cor : integer;
begin
    
    -- Finite State Machine
    -- current state logic 
    process(clk,reset)
    begin
        if (reset = '1') then
            curr_state <= idle;
        elsif (clk'event and clk = '1') then
            curr_state <= next_state;
        end if;
    end process;
    
    -- next state logic 
    process(curr_state)
    begin
        case curr_state is
            when idle => 
                if ready = '0' then
                    next_state <= idle;
                else
                    next_state <= senddata;
                    x_cor <= x_min;
                    y_cor <= y_min;
                end if;
            when senddata =>
                if x_cor < x_max then
                    x_cor <= x_cor + 1;
                    next_state <= senddata;
                else
                    x_cor <= x_min;
                    next_state <= decision;
                end if;
            when decision =>
                if y_cor < y_max then
                    y_cor <= y_cor + 1;
                    next_state <= senddata;
                else
                    y_cor <= y_min;
                    next_state <= idle;
                end if;
        end case;
        
   end process;
   -- moore output
   
   
   -- mealy output

end Behavioral;
