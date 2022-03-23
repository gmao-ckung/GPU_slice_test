program slice_test

    implicit none

    double precision, allocatable, dimension(:,:,:) :: A, B, C
    double precision :: constant

    character * 100 :: inputBuffer
    integer :: i_size, j_size, k_size, ii, jj, kk, iter, iterT
    real :: start_time, end_time

    call getarg(1, inputBuffer)
    read(inputBuffer,*) i_size
    call getarg(2, inputBuffer)
    read(inputBuffer,*) j_size
    call getarg(3, inputBuffer)
    read(inputBuffer,*) k_size

    write(*,*) 'i_size = ', i_size
    write(*,*) 'j_size = ', j_size
    write(*,*) 'k_size = ', k_size


    allocate(A(i_size, j_size, k_size))
    allocate(B(i_size, j_size, k_size))
    allocate(C(i_size, j_size, k_size))

    A = 1.0
    B = 2.0
    C = 0.0

    constant = 3.0

    iterT = 1000

!$acc data copyin(A, B, C, constant)

    call cpu_time(start_time)
!$acc parallel present(A,B,C,constant)
    C = A + constant*B    
!$acc end parallel
    call cpu_time(end_time)
    
    write(*,*) 'Time with implied array computation = ', end_time - start_time
    
!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)  

    call cpu_time(start_time)
!$acc parallel present(A, B, C, constant)    
    do iter = 1,iterT
        do kk = 1,k_size
            do jj = 1,j_size
                do ii = 1,i_size
                    C(ii,jj,kk) = A(ii,jj,kk) + constant*B(ii,jj,kk)
                enddo
            enddo
        enddo
    enddo
!$acc end parallel
    call cpu_time(end_time)

    write(*,*) 'Time only with Parallel = ', end_time - start_time
    
!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)    

    call cpu_time(start_time)
!$acc parallel present(A, B, C, constant)
    do iter = 1,iterT
    !$acc loop 
        do kk = 1,k_size
            C(:,:,kk) = A(:,:,kk) + constant*B(:,:,kk)
            ! A(:,:,kk) = B(:,:,kk) + constant*C(:,:,kk)
            ! B(:,:,kk) = C(:,:,kk) + constant*A(:,:,kk)
        enddo
    enddo
!$acc end parallel
    call cpu_time(end_time)

    write(*,*) 'Time with 2 slices = ', end_time - start_time

!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)

    call cpu_time(start_time)
!$acc parallel present(A, B, C, constant)
    do iter = 1,iterT
    !$acc loop
        do kk = 1,k_size
            do jj = 1,j_size
                C(:,jj,kk) = A(:,jj,kk) + constant*B(:,jj,kk)
                ! A(:,jj,kk) = B(:,jj,kk) + constant*C(:,jj,kk)
                ! B(:,jj,kk) = C(:,jj,kk) + constant*A(:,jj,kk)
            enddo
        enddo
    enddo
!$acc end parallel
    call cpu_time(end_time)

    write(*,*) 'Time with 1 slice = ', end_time - start_time

!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)

    call cpu_time(start_time)
!$acc parallel present(A, B, C, constant)
    do iter = 1,iterT
    !$acc loop
        do kk = 1,k_size
            do jj = 1,j_size
                do ii = 1,i_size
                    C(ii,jj,kk) = A(ii,jj,kk) + constant*B(ii,jj,kk)
                    ! A(ii,jj,kk) = B(ii,jj,kk) + constant*C(ii,jj,kk)
                    ! B(ii,jj,kk) = C(ii,jj,kk) + constant*A(ii,jj,kk)
                enddo
            enddo
        enddo
    enddo
!$acc end parallel
    call cpu_time(end_time)

    write(*,*) 'Time with no slices = ', end_time - start_time

!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)

    call cpu_time(start_time)
!$acc parallel present(A, B, C, constant)
    do iter = 1,iterT
    !$acc loop collapse(2)
        do kk = 1,k_size
            do jj = 1,j_size
                C(:,jj,kk) = A(:,jj,kk) + constant*B(:,jj,kk)
                ! A(:,jj,kk) = B(:,jj,kk) + constant*C(:,jj,kk)
                ! B(:,jj,kk) = C(:,jj,kk) + constant*A(:,jj,kk)
            enddo
        enddo
    enddo
!$acc end parallel
    call cpu_time(end_time)

    write(*,*) 'Time with 1 slice and collapsed looping = ', end_time - start_time

!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)

    call cpu_time(start_time)
!$acc parallel present(A, B, C, constant)
    do iter=1,iterT
    !$acc loop collapse(3)
        do kk = 1,k_size
            do jj = 1,j_size
                do ii = 1,i_size
                    C(ii,jj,kk) = A(ii,jj,kk) + constant*B(ii,jj,kk)
                    ! A(ii,jj,kk) = B(ii,jj,kk) + constant*C(ii,jj,kk)
                    ! B(ii,jj,kk) = C(ii,jj,kk) + constant*A(ii,jj,kk)
                enddo
            enddo
        enddo
    enddo
!$acc end parallel
    call cpu_time(end_time)

    write(*,*) 'Time with no slices and collapsed looping = ', end_time - start_time

!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)

    call cpu_time(start_time)
    !$acc kernels present(A,B,C)
    do iter=1,iterT
        do kk = 1,k_size
            do jj = 1,j_size
                do ii = 1,i_size
                    C(ii,jj,kk) = A(ii,jj,kk) + constant*B(ii,jj,kk)
                    ! A(ii,jj,kk) = B(ii,jj,kk) + constant*C(ii,jj,kk)
                    ! B(ii,jj,kk) = C(ii,jj,kk) + constant*A(ii,jj,kk)
                enddo
            enddo
        enddo
    enddo
    !$acc end kernels
    call cpu_time(end_time)

    write(*,*) 'Time with kernels statement = ', end_time - start_time

!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)

    call cpu_time(start_time)
!$acc parallel present(A, B, C, constant)
    do iter=1,iterT
    !$acc loop gang collapse(2)
        do kk = 1,k_size
            do jj = 1,j_size
    !$acc loop vector
                do ii = 1,i_size
                    C(ii,jj,kk) = A(ii,jj,kk) + constant*B(ii,jj,kk)
                    ! A(ii,jj,kk) = B(ii,jj,kk) + constant*C(ii,jj,kk)
                    ! B(ii,jj,kk) = C(ii,jj,kk) + constant*A(ii,jj,kk)
                enddo
            enddo
        enddo
    enddo
!$acc end parallel
    call cpu_time(end_time)

    write(*,*) 'Time with gang collapse(2) and vector statements = ', end_time - start_time

!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)

    call cpu_time(start_time)
!$acc parallel present(A, B, C, constant)
    do iter=1,iterT
    !$acc loop gang
        do kk = 1,k_size
    !$acc loop worker
            do jj = 1,j_size
    !$acc loop vector
                do ii = 1,i_size
                    C(ii,jj,kk) = A(ii,jj,kk) + constant*B(ii,jj,kk)
                    ! A(ii,jj,kk) = B(ii,jj,kk) + constant*C(ii,jj,kk)
                    ! B(ii,jj,kk) = C(ii,jj,kk) + constant*A(ii,jj,kk)
                enddo
            enddo
        enddo
    enddo
!$acc end parallel
    call cpu_time(end_time)

    write(*,*) 'Time with gang, worker, and vector statements = ', end_time - start_time

!$acc update host(C)
    write(*,*) "sum(C) = ", sum(C)

    call cpu_time(start_time)
    do iter=1,iterT
        do concurrent (ii = 1:i_size, jj = 1:j_size, kk = 1:k_size)
            C(ii,jj,kk) = A(ii,jj,kk) + constant*B(ii,jj,kk)
        enddo
    enddo
    call cpu_time(end_time)

    write(*,*) 'Time with Fortran do concurrent = ', end_time - start_time
!$acc update host(c)
    write(*,*) "sum(C) = ", sum(C)

!$acc end data
end program