

x=array(float,256,128,128);
s=readsnap("runs2/out/snap.00043.p00000",nbnd=16);
x(:128,,)=(*s.xion);
s=readsnap("runs2/out/snap.00043.p00001",nbnd=16);
x(129:,,)=(*s.xion);
