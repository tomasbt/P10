# create EXP ROM code

import numpy as np


def convFixToStr(val, q=7):
    '''
    Will take a fixed-point value and convert to a string of the binary value
    '''
    tmpstr = ''
    for i in range(q, -1, -1):
        tmpstr = tmpstr + str((val >> i) & 1)

    return tmpstr

if __name__ == '__main__' or True:
    n = 256
    sz = 15
    sig = 25.0
    s = '   type ROM_Array is array (0 to ' + str(n - 1) + ')'
    print s
    s = '       of std_logic_vector (' + str(sz - 1) + ' downto 0);'
    print s
    print ''

    print '   constant Content: ROM_Array := ('
    for i in range(n-1):
        expVal = np.exp(-i / sig)
        tmp = int(expVal * 2**(sz - 1)) / 2.0**(sz - 1)
        strVal = convFixToStr(int(expVal * 2**(sz - 1)), q=sz - 1)
        print '       ' + str(i) + ' => "' + strVal + '",'
    print '       OTHERS => "'+convFixToStr(0, q=sz - 1)+'"'
    print '       );'
    print '\n--------------\n'
    # after begin
    print '   expTabel : process(Clock, Reset, Read, Address)'
    print '   begin'
    print '       if( Reset = \'1\' ) then'
    tmpstr = ''
    for i in range(sz):
        tmpstr = tmpstr + 'Z'
    print '           Data_out <= "'+tmpstr+'";'
    print '       elsif( Clock\'event and Clock = \'1\' ) then'
    print '           if Enable = \'1\' then'
    print '               if( Read = \'1\' ) then'
    print '                   Data_out <= Content(conv_integer(Address));'
    print '               else'
    print '                   Data_out <= "'+tmpstr+'";'
    print '               end if;'
    print '           end if;'
    print '       end if;'
    print '   end process;'
