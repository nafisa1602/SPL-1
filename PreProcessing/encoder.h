#ifndef ENCODER_H
#define ENCODER_H
int charToIndex(char c)
{
    if (c >= 'a' && c <= 'z') return (c - 'a') + 1;
    if (c >= '0' && c <= '9') return (c - '0') + 27;
    if (c == '.') return 37;
    if (c == '-') return 38;
    if (c == '_') return 39;
    return 0;
}
#endif