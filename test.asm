 push    r15
 push    r14
 push    r13
 push    r12
 push    rsi
 push    rdi
 push    rbp
 push    rbx
 sub     rsp, 248
 mov     r14, rdx
 mov     rax, rcx
 mov     r10, qword, ptr, [rcx, +, 88]
 mov     edx, 1
 mov     ecx, r10d
 shl     rdx, cl
 cmp     rdx, r8
 jne     .PANIC
 mov     rbp, qword, ptr, [r9, +, 16]
 cmp     rbp, qword, ptr, [rax, +, 64]
 jne     .PANIC
 test    r10, r10
 je      .FINISH
 mov     rsi, qword, ptr, [r9]
 mov     r13, qword, ptr, [r9, +, 8]
 mov     rcx, qword, ptr, [r9, +, 24]
 mov     qword, ptr, [rsp, +, 48], rcx
 mov     rcx, qword, ptr, [rax, +, 24]
 mov     qword, ptr, [rsp, +, 64], rcx
 mov     r12, qword, ptr, [rax, +, 40]
 lea     rax, [rbp, +, rbp]
 mov     qword, ptr, [rsp, +, 56], rax
 lea     rax, [r8, +, 3]
 mov     qword, ptr, [rsp, +, 176], rax
 xor     edx, edx
 mov     qword, ptr, [rsp, +, 104], r14
 mov     qword, ptr, [rsp, +, 160], rbp
 mov     qword, ptr, [rsp, +, 152], r13
 mov     qword, ptr, [rsp, +, 168], r10
.OUTER_LOOP:
 mov     eax, edx
 mov     qword, ptr, [rsp, +, 184], rdx
 and     edx, 63
 mov     r9d, 1
 mov     ecx, edx
 shl     r9, cl
 mov     qword, ptr, [rsp, +, 112], r9
 not     eax
 add     eax, r10d
 mov     r15d, 1
 mov     ecx, eax
 shl     r15, cl
 mov     eax, 2
 shl     rax, cl
 mov     r10, -2
 mov     ecx, edx
 shl     r10, cl
 cmp     rdx, 2
 mov     qword, ptr, [rsp, +, 96], r15
 mov     rcx, rax
 mov     qword, ptr, [rsp, +, 40], rax
 jae     .LBB31_5
 add     r10, r8
 xor     ecx, ecx
 mov     qword, ptr, [rsp, +, 72], r10
 jmp     .LBB31_10
.LBB31_9:
 mov     rcx, qword, ptr, [rsp, +, 80]
 cmp     rcx, r15
 mov     r10, qword, ptr, [rsp, +, 72]
 je      .LBB31_66
.LBB31_10: 
 mov     qword, ptr, [rsp, +, 88], rcx
 inc     rcx
 mov     qword, ptr, [rsp, +, 80], rcx
 mov     rcx, r10
 xor     r10d, r10d
 jmp     .LBB31_11
.LBB31_22: ; inner loop?
 mov     r8, r12
 mov     r15, qword, ptr, [rsp, +, 96]
 mov     rax, qword, ptr, [rsp, +, 40]
 mov     qword, ptr, [r14, +, 8*rdi], r9
 inc     rcx
 cmp     r10, qword, ptr, [rsp, +, 112]
 mov     r12, r11
 mov     rbp, qword, ptr, [rsp, +, 160]
 mov     r13, qword, ptr, [rsp, +, 152]
 je      .LBB31_9
.LBB31_11:
 cmp     rcx, r12
 jae     .PANIC
 mov     rbx, r10
 imul    rbx, rax
 add     rbx, qword, ptr, [rsp, +, 88]
 cmp     rbx, r8
 jae     .PANIC
 lea     rdi, [rbx, +, r15]
 cmp     rdi, r8
 jae     .PANIC
 mov     r11, r12
 mov     r12, r8
 mov     rax, qword, ptr, [rsp, +, 64]
 mov     rax, qword, ptr, [rax, +, 8*rcx]
 mov     r15, qword, ptr, [r14, +, 8*rbx]
 mul     qword, ptr, [r14, +, 8*rdi]
 mov     r9, rdx
 mov     r8, rax
 mov     r14, rax
 imul    r14, r13
 mul     rsi
 imul    r9, rsi
 add     r9, rdx
 add     r9, r14
 shr     r9, 20
 imul    r9, rbp
 sub     r8, r9
 lea     r9, [r15, +, r8]
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_20
 mov     rax, rsi
 mul     r9
 imul    r13, r9
 add     r13, rdx
 shr     r13, 20
 imul    r13, rbp
 sub     r9, r13
.LBB31_20:
 inc     r10
 mov     r14, qword, ptr, [rsp, +, 104]
 mov     qword, ptr, [r14, +, 8*rbx], r9
 mov     r9, qword, ptr, [rsp, +, 56]
 sub     r9, r8
 add     r9, r15
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_22
 mov     rax, rsi
 mul     r9
 mov     rax, qword, ptr, [rsp, +, 152]
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, qword, ptr, [rsp, +, 160]
 sub     r9, rax
 jmp     .LBB31_22
.LBB31_5:
 add     r10, qword, ptr, [rsp, +, 176]
 mov     rax, qword, ptr, [rsp, +, 40]
 lea     rax, [8*rax]
 lea     rax, [rax, +, 2*rax]
 lea     rcx, [r14, +, rax]
 mov     qword, ptr, [rsp, +, 128], rcx
 mov     rcx, qword, ptr, [rsp, +, 40]
 lea     rcx, [rcx, +, 2*rcx]
 lea     rax, [rax, +, 8*r15]
 add     rax, r14
 mov     qword, ptr, [rsp, +, 120], rax
 mov     qword, ptr, [rsp, +, 208], rcx
 lea     rax, [r15, +, rcx]
 mov     qword, ptr, [rsp, +, 200], rax
 mov     rax, qword, ptr, [rsp, +, 40]
 shl     rax, 4
 lea     rcx, [r14, +, rax]
 mov     qword, ptr, [rsp, +, 144], rcx
 mov     rcx, qword, ptr, [rsp, +, 40]
 add     rcx, rcx
 mov     qword, ptr, [rsp, +, 224], rcx
 lea     rcx, [rax, +, 8*r15]
 mov     rax, qword, ptr, [rsp, +, 40]
 add     rcx, r14
 mov     qword, ptr, [rsp, +, 136], rcx
 lea     rcx, [r15, +, 2*rax]
 mov     qword, ptr, [rsp, +, 216], rcx
 lea     rcx, [r14, +, 8*rax]
 mov     qword, ptr, [rsp, +, 88], rcx
 add     rax, r15
 mov     qword, ptr, [rsp, +, 232], rax
 lea     rax, [r14, +, 8*rax]
 mov     qword, ptr, [rsp, +, 80], rax
 lea     rax, [r14, +, 8*r15]
 mov     qword, ptr, [rsp, +, 240], rax
 xor     r11d, r11d
 mov     qword, ptr, [rsp, +, 72], r10
 jmp     .LBB31_24
.LBB31_23:
 mov     r11, qword, ptr, [rsp, +, 192]
 cmp     r11, r15
 mov     r14, qword, ptr, [rsp, +, 104]
 je      .LBB31_66
.LBB31_24:
 lea     rax, [r11, +, 1]
 mov     qword, ptr, [rsp, +, 192], rax
 xor     r14d, r14d
 mov     rbx, r10
 xor     eax, eax
 jmp     .LBB31_25
.LBB31_65:
 mov     r10, qword, ptr, [rsp, +, 72]
 mov     rdx, qword, ptr, [rsp, +, 120]
 mov     qword, ptr, [rdx, +, 8*r11], r9
 add     rcx, rax
 add     rbx, 4
 mov     al, 1
 mov     r11, rcx
.LBB31_25:
 test    al, 1
 je      .LBB31_29
 add     r14, 3
 jb      .LBB31_23
 cmp     r14, qword, ptr, [rsp, +, 112]
 jae     .LBB31_23
 inc     r14
 jmp     .LBB31_30
.LBB31_29:
 mov     rcx, qword, ptr, [rsp, +, 112]
 cmp     r14, rcx
 mov     rax, r14
 adc     rax, 0
 cmp     r14, rcx
 mov     r14, rax
 jae     .LBB31_23
.LBB31_30:
 lea     rax, [rbx, -, 3]
 cmp     rax, r12
 jae     .PANIC
 cmp     r11, r8
 jae     .PANIC
 lea     rax, [r15, +, r11]
 cmp     rax, r8
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 64]
 mov     rax, qword, ptr, [rax, +, 8*rbx, -, 24]
 mov     rcx, qword, ptr, [rsp, +, 104]
 mov     r10, qword, ptr, [rcx, +, 8*r11]
 mov     r15, qword, ptr, [rsp, +, 240]
 mul     qword, ptr, [r15, +, 8*r11]
 mov     r9, rdx
 mov     rcx, rax
 mov     rdi, rax
 imul    rdi, r13
 mul     rsi
 imul    r9, rsi
 add     r9, rdx
 add     r9, rdi
 shr     r9, 20
 imul    r9, rbp
 sub     rcx, r9
 lea     r9, [r10, +, rcx]
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_41
 mov     rax, rsi
 mul     r9
 mov     rax, r13
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, rbp
 sub     r9, rax
.LBB31_41:
 mov     rax, qword, ptr, [rsp, +, 104]
 mov     qword, ptr, [rax, +, 8*r11], r9
 mov     r9, qword, ptr, [rsp, +, 56]
 sub     r9, rcx
 add     r9, r10
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_43
 mov     rax, rsi
 mul     r9
 mov     rax, r13
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, rbp
 sub     r9, rax
.LBB31_43:
 mov     qword, ptr, [r15, +, 8*r11], r9
 lea     rax, [rbx, -, 2]
 cmp     rax, r12
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 40]
 lea     rcx, [r11, +, rax]
 cmp     rcx, r8
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 232]
 add     rax, r11
 cmp     rax, r8
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 64]
 mov     rax, qword, ptr, [rax, +, 8*rbx, -, 16]
 mov     rdx, qword, ptr, [rsp, +, 88]
 mov     r10, qword, ptr, [rdx, +, 8*r11]
 mov     rdx, qword, ptr, [rsp, +, 80]
 mul     qword, ptr, [rdx, +, 8*r11]
 mov     r9, rdx
 mov     rdi, rax
 mov     r15, rax
 imul    r15, r13
 mul     rsi
 imul    r9, rsi
 add     r9, rdx
 add     r9, r15
 shr     r9, 20
 imul    r9, rbp
 sub     rdi, r9
 lea     r9, [r10, +, rdi]
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_48
 mov     rax, rsi
 mul     r9
 mov     rax, r13
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, rbp
 sub     r9, rax
.LBB31_48:
 mov     rax, qword, ptr, [rsp, +, 88]
 mov     qword, ptr, [rax, +, 8*r11], r9
 mov     r9, qword, ptr, [rsp, +, 56]
 sub     r9, rdi
 add     r9, r10
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_50
 mov     rax, rsi
 mul     r9
 mov     rax, r13
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, rbp
 sub     r9, rax
.LBB31_50:
 mov     rax, qword, ptr, [rsp, +, 80]
 mov     qword, ptr, [rax, +, 8*r11], r9
 lea     rax, [rbx, -, 1]
 cmp     rax, r12
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 224]
 add     rax, r11
 cmp     rax, r8
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 216]
 add     rax, r11
 cmp     rax, r8
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 64]
 mov     rax, qword, ptr, [rax, +, 8*rbx, -, 8]
 mov     rdx, qword, ptr, [rsp, +, 144]
 mov     r10, qword, ptr, [rdx, +, 8*r11]
 mov     rdx, qword, ptr, [rsp, +, 136]
 mul     qword, ptr, [rdx, +, 8*r11]
 mov     r9, rdx
 mov     rdi, rax
 mov     r15, rax
 imul    r15, r13
 mul     rsi
 imul    r9, rsi
 add     r9, rdx
 add     r9, r15
 shr     r9, 20
 imul    r9, rbp
 sub     rdi, r9
 lea     r9, [r10, +, rdi]
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_55
 mov     rax, rsi
 mul     r9
 mov     rax, r13
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, rbp
 sub     r9, rax
.LBB31_55:
 mov     r15, qword, ptr, [rsp, +, 96]
 mov     rax, qword, ptr, [rsp, +, 144]
 mov     qword, ptr, [rax, +, 8*r11], r9
 mov     r9, qword, ptr, [rsp, +, 56]
 sub     r9, rdi
 add     r9, r10
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_57
 mov     rax, rsi
 mul     r9
 mov     rax, r13
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, rbp
 sub     r9, rax
.LBB31_57:
 mov     rax, qword, ptr, [rsp, +, 136]
 mov     qword, ptr, [rax, +, 8*r11], r9
 cmp     rbx, r12
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 208]
 add     rax, r11
 cmp     rax, r8
 jae     .PANIC
 mov     rax, qword, ptr, [rsp, +, 200]
 add     rax, r11
 cmp     rax, r8
 jae     .PANIC
 add     rcx, qword, ptr, [rsp, +, 40]
 mov     rax, qword, ptr, [rsp, +, 64]
 mov     rax, qword, ptr, [rax, +, 8*rbx]
 mov     rdx, qword, ptr, [rsp, +, 120]
 mul     qword, ptr, [rdx, +, 8*r11]
 mov     r9, rdx
 mov     rdi, rax
 mul     rsi
 imul    r9, rsi
 add     r9, rdx
 mov     rax, rdi
 imul    rax, r13
 add     r9, rax
 mov     rax, qword, ptr, [rsp, +, 128]
 mov     r10, qword, ptr, [rax, +, 8*r11]
 shr     r9, 20
 imul    r9, rbp
 sub     rdi, r9
 lea     r9, [r10, +, rdi]
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_63
 mov     rax, rsi
 mul     r9
 mov     rax, r13
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, rbp
 sub     r9, rax
.LBB31_63:
 mov     rax, qword, ptr, [rsp, +, 40]
 add     rcx, rax
 mov     rdx, qword, ptr, [rsp, +, 128]
 mov     qword, ptr, [rdx, +, 8*r11], r9
 mov     r9, qword, ptr, [rsp, +, 56]
 sub     r9, rdi
 add     r9, r10
 cmp     r9, qword, ptr, [rsp, +, 48]
 jbe     .LBB31_65
 mov     rax, rsi
 mul     r9
 mov     rax, r13
 imul    rax, r9
 add     rax, rdx
 shr     rax, 20
 imul    rax, rbp
 sub     r9, rax
 mov     rax, qword, ptr, [rsp, +, 40]
 jmp     .LBB31_65
.LBB31_66:
 mov     rdx, qword, ptr, [rsp, +, 184]
 inc     rdx
 mov     r10, qword, ptr, [rsp, +, 168]
 cmp     rdx, r10
 jne     .OUTER_LOOP
.FINISH:
 add     rsp, 248
 pop     rbx
 pop     rbp
 pop     rdi
 pop     rsi
 pop     r12
 pop     r13
 pop     r14
 pop     r15
 ret