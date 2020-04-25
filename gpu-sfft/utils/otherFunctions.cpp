

// this modInverse code is copied from https://comeoncodeon.wordpress.com/2011/10/09/modular-multiplicative-inverse/
int modInverse(int a, int mod) {
    int b = mod-2;

    int x = 1, y = a;
    while(b > 0) {
        if(b%2 == 1) {
            x=(x*y);
            if(x>mod) x%=mod;
        }
        y = (y*y);
        if(y>mod) y%=mod;
        b /= 2;
    }
    return x;
}