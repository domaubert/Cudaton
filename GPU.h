#define NCELLS 128
#define NCELLX (NCELLS)
#define NCELLY (NCELLS)
#define NCELLZ (NCELLS)
#define NBUFF (NCELLZ) // MUST BE the MAX of NCELLX,Y,Z

#define NGPUX 2 
#define NGPUY 1
#define NGPUZ 1

#define BLOCKCOOL 128 //MUST BE <128
#define GRIDCOOLX 4 // can be increased (4 for 256^3)
#define GRIDCOOLY ((NCELLX*NCELLY*NCELLZ)/GRIDCOOLX/BLOCKCOOL) 
