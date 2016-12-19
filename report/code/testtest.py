print "calculate EEPSM complexity"
L = 23
om = 8
d = 303
m = 2056
n = 2464
add = ((12 + om) * d + 12 * L)
mul = (8 * d + 12 * (L * (L - 1) / 2))
div = 0
comp = om * (d + 1)
print "C Add", add+comp, " mul", mul, " div", div, " comp", comp
L = 23
om = 8
d = 303
m = 2056
n = 2464
add = ((8 + om) * d + 4 * L)
mul = (7 * d + 4 * (L * (L - 1) / 2))
div = 0
comp = om * (d + 1)
print "G Add", add+comp, " mul", mul, " div", div, " comp", comp


print "calculate FCV complexity"
r = 9
d = 303
m = 2056
n = 2464
add = 977 * d
mul = 210 * d
div = d
comp = 0
print "C Add", add, " mul", mul, " div", div, " comp", comp

r = 9
d = 303
m = 2056
n = 2464
add = 346 * d
mul = 53 * d
div = d
comp = 0
print "G Add", add, " mul", mul, " div", div, " comp", comp

om = 25
d = 303
m = 2056
n = 2464
read = (39 + 21 * d)
print 'mem use EESPM C   reads', read

om = 25
d = 303
m = 2056
n = 2464
read = (21 + 19 * d)
print 'mem use EESPM G   reads', read

r = 9
d = 303
m = 2056
n = 2464
read = (44 + 32 * d)
print 'mem use fcv C   reads', read

r = 9
d = 303
m = 2056
n = 2464
read = (13 + 16 * d)
print 'mem use fcv G   reads', read

om = 25
d = 303
m = 2056
n = 2464
save = (4 + 9 * d)
print 'mem use EESPM C   save', save

om = 25
d = 303
m = 2056
n = 2464
save = (4 + 9 * d)
print 'mem use EESPM G   save', save

r = 9
d = 303
m = 2056
n = 2464
save = (9 + 19 * d)
print 'mem use fcv C   save', save

r = 9
d = 303
m = 2056
n = 2464
save = (3 + 10 * d)
print 'mem use fcv G   save', save
