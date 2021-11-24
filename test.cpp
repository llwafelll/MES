#include<iostream>
#include<stdlib.h>

int **mktab(int n, int m) {
    int **tab = new int*[m];
    for (int i = 0; i < m; ++i)
        tab[i] = new int[m];
}
int main() {
    std::cout << "Hello wordl";
    return 0;
}