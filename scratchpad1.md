mu = -.3
std= .1
[-.4, -.2]


mouse-x: ~ -.3 kiri  [-1, 1]*10
y: ~ -.3 atas

mu1, sigma1 == mouse x
mu2, sigma2 == mouse y
y == mouse down


> Experiment #1: 
  Nilai delta dari critic value diberikan constrain berupa suatu konstanta
  Tujuan: membandingkan tanpa konstanta vs ada konstanta
  
  Rumusan masalah:
    Apakah dengan menambahkan konstanta pada loss akan meningkatkan success rate?

Experiment #2: 
  Agent mengatur step size nya sendiri