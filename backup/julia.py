
Jam  1:31 PM
We were once profiling a matrix multiplication in numpy that was unbeatalbe no matter what we did.
However, I vaguely recall you ended up trying to reproduce it in Julia, and found Julia gave a 4 timesish speed up, maybe. Do you have any recollection of this, and know if you have that Julia code?
That one function is now the kill bottleneck of everything interferometry
1:33
Numpy linear algebra must have god-like multiproessing as it speeds up a lot over 20 CPUs compared to 1.

Jam  1:44 PM
On a related note, we have a function which preloads various aspects of the computation and outperforms numpy for a 1 CPU problem.
However, Numpy wins for > 10 CPU by a lot -- its linear algebra calculation goes into multiprocessing mode in a way our own function cannot win. I had a go at multitrheading / parallelizing our function via numba, but it was a complete non starter.
I am considering coding our function up in Fortran (or C, but I'm more faimilar with Fortran). Do you know if getting it to multiprocessing requires a lot of care and effort, or is it something the compiler might just do by "magic"

Jonathan Frawley  2:31 PM
We were once profiling a matrix multiplication in numpy that was unbeatalbe no matter what we did.
However, I vaguely recall you ended up trying to reproduce it in Julia, and found Julia gave a 4 timesish speed up, maybe. Do you have any recollection of this, and know if you have that Julia code?
I will have a look now and let you know.
 I am considering coding our function up in Fortran (or C, but I'm more faimilar with Fortran). Do you know if getting it to multiprocessing requires a lot of care and effort, or is it something the compiler might just do by "magic"
I think the main options for both Fortran and C are very similar: use OpenMP for single-node parallelisation and MPI for multi-node parallelisation. No compiler will spread the code out over multiple cores / nodes magically in those languages unfortunately. The compiler can do lots of single-threaded optimisations for you but that's it.

Jam  2:34 PM
Damn, I would imagine the conclusion is the preloading calculation is great for single CPU but is not suitable for MPI parallelization (hard to avoid passing large arrays of data).

Jonathan Frawley  3:34 PM
In that case I think using C / Fortran would be the best way to speed it up alright. If it involves a large amount of data, things like Vectorization will help there actually (which the Intel compiler will automatically do if you write your code the right way). There is also CUDA which could help but is much more tricky to get right.
3:35
On the Numpy / Julia code, I found the example I was using:
import time

import numpy as np


if __name__ == '__main__':
    a = np.random.rand(2048, 2048)
    b = np.random.rand(2048, 2048)
    c = np.zeros((2048, 2048))

    iters = 100

    start = time.time()
    for i in range(iters):
        c = np.dot(a, b)
    end = time.time()

    print(f'np time: {end-start}')
 (edited)
3:35
That's the Python benchmark which takes ~ 6.5 secs on my machine.
3:36
This Julia program has 2 different methods:
function matmul(a, b)
  return a * b
end

function matmul_inline(a, b, c)
  @. c = a * b
end

function main()
  a = rand(Float64, (2048, 2048))
  b = rand(Float64, (2048, 2048))
  c = rand(Float64, (2048, 2048))

  @time for i = 1:100
    c = matmul(a, b)
  end

  @time for i = 1:100
    matmul_inline(a, b, c)
  end
end

main()
 (edited)
3:37
The "matmul_inline" benchmark takes 0.494965 seconds, the non-inline version takes 6.524583 seconds which is the same as numpy, which makes sense.
3:38
So quite a bit faster, assuming you can preallocate your matrices.

Jam  4:00 PM
 In that case I think using C / Fortran would be the best way to speed it up alright. If it involves a large amount of data, things like Vectorization will help there actually (which the Intel compiler will automatically do if you write your code the right way). There is also CUDA which could help but is much more tricky to get right.
Yeah, my worry is its a lot of time investment and feels like it might not quite come together. I am tempted to give it a shot, but will probably exhaust all other options first.
4:02
 The "matmul_inline" benchmark takes 0.494965 seconds, the non-inline version takes 6.524583 seconds which is the same as numpy, which makes sense.
[3:38 PM] So quite a bit faster, assuming you can preallocate your matrices.
If Julia has similar inbuilt multiprocessing capabilities as numpy, such that we get similar scaling as a function of CPU #, this is a bit of a game changer. This definitely feels like the next avenue for us to pursue, x10 would be rediuclous.
Do you have any advise / experience on whether wrapping the Julia code in Python and running on cosma would have any issues?
I am going to also try JAX, if you've heard of that.
4:03
Also, what is the difference between matmul and matmul_inline?

Jonathan Frawley  4:16 PM
Sorry, I should have called the function "matmul_inplace" rather than inline, here is that updated:
function matmul(a, b)
  return a * b
end

function matmul_inplace(a, b, c)
  @. c = a * b
end

function main()
  a = rand(Float64, (2048, 2048))
  b = rand(Float64, (2048, 2048))
  c = rand(Float64, (2048, 2048))

  @time for i = 1:100
    c = matmul(a, b)
  end

  @time for i = 1:100
    matmul_inplace(a, b, c)
  end
end

main()
So, the Julia "matmul" function is the exact analogue of the numpy "dot" function - it returns a newly allocated matrix. "matmul_inplace" uses the passed in matrix "c" and uses that to store the result. I think this a key reason why the Julia version is so much faster - allocating that much memory takes a lot of time. Numpy has a way of doing a matrix multipication inline as well but it doesn't result in a significant speedup, you can try it out here:
import time

import numpy as np


if __name__ == '__main__':
    a = np.random.rand(2048, 2048)
    b = np.random.rand(2048, 2048)
    c = np.zeros((2048, 2048))

    iters = 100

    start = time.time()
    for i in range(iters):
        c = np.dot(a, b)
    end = time.time()

    print(f'np time: {end-start}')

    start = time.time()
    for i in range(iters):
        np.matmul(a, b, out=c)
    end = time.time()

    print(f'np inline time: {end-start}')
Here are the results of that:
np time: 4.876985788345337
np inline time: 3.631227731704712
Not really sure why the numpy inline method is so much slower than Julia's but I am guessing it has something to do with the fact that very few people use numpy in that way.
4:17
If Julia has similar inbuilt multiprocessing capabilities as numpy, such that we get similar scaling as a function of CPU #, this is a bit of a game changer. This definitely feels like the next avenue for us to pursue, x10 would be rediuclous.
My understanding is that they both use BLAS under the hood for these operations, so they should scale similarly.
4:18
I haven't used JAX but it sounds interesting.

Jam  4:20 PM
Any insight of the Python / Cosma wrapping? Got some other folk I could ask on this.
4:20
 So, the Julia "matmul" function is the exact analogue of the numpy "dot" function - it returns a newly allocated matrix. "matmul_inplace" uses the passed in matrix "c" and uses that to store the result. I think this a key reason why the Julia version is so much faster - allocating that much memory takes a lot of time. Numpy has a way of doing a matrix multipication inline as well but it doesn't result in a significant speedup, you can try it out here:
Makes sense, the speed up is utterly insane... We will be trying this almost instantly haha

Jonathan Frawley  4:21 PM
Here you go:
import julia
import time

j = julia.Julia()


start = time.time()
fn = j.include('main.jl')

fn
end = time.time()
print(f'time taken: {end-start}')

Jam  4:21 PM
 I haven't used JAX but it sounds interesting.
I did some googling and it sounds like it wont speed up this problem (maybe I should check for an _inplace function) . But the autograd stuff is mindblowing. AutoLens will be going JAX one day...

Jonathan Frawley  4:22 PM
Sorry, was just trying it on COSMA, it works okay but has a bit of overhead of ~2s.
4:22
I think there is a way to integrate it that would be faster, that is just the quick and dirty way.

Jam  4:22 PM
It currrently takes > 60 seconds :sweat_smile:
:slightly_smiling_face:
1


Jonathan Frawley  4:25 PM
Here is more info about the Julia -> Python bridge: https://pyjulia.readthedocs.io/en/latest/usage.html

Jam  4:25 PM
Amazing thank you. Seriously would be rediculous if this random investigation of yours from half a year ago came together!
4:26
Will let you know how we get on!

Jonathan Frawley  4:27 PM
No problem! Hopefully it works out. Yes, please let me know how you get on. I would be happy to take a look at a particular matrix multiplication if you can give me the raw data as well.
4:29
Just of note on COSMA, there is a julia module to load:
module load julia
then you need to install the Python julia package:
pip3 install --user julia
and then init that:
python3
>>> import julia
>>> julia.install()
Then you should be able to run the above.
:+1:
1


Jam  4:29 PM
 I would be happy to take a look at a particular matrix multiplication if you can give me the raw data as well.
We'll probably take you up on that offer -- will give it a go ourselves first.
4:31
Final question, do you know if other Python linear algebra methods (cholesky, solve) have inplace methods (in numpy but I would guess more importantly in Julia, as this could offer some nice speed up elsewhere).

Jonathan Frawley  4:45 PM
It looks like the scipy version of cholesky and solve have an overwrite_a and overwrite_b options which let you do those in-place:
cholesky https://stackoverflow.com/a/14408981
linalg.solve: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html
I don't think the numpy versions have an equivalent however.
Just checking Julia now.

Jonathan Frawley  4:59 PM
There is an in-place version of cholesky in Julia (anything ending in a "!" is in place in Julia): https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.cholesky!! (edited)

Jonathan Frawley  5:05 PM
And the same for solve (which is just the \ operator in Julia. You need to start the line of code with @. to say that it is in place:
using LinearAlgebra
function solve(a, b)
  return a \ b
end

function solve_inplace(a,b )
   @. a \ b
end

function main()
  a = [1 0; 1 -2]
  b = [32; -4]


  @time for i = 1:100
    solve(a, b)
  end

  @time for i = 1:100
    solve_inplace(a, b)
  end
end

main()
5:06
And cholesky:
using LinearAlgebra


function cho(a)
  return cholesky(a)
end

function cho_inplace(a)
  return cholesky!(a)
end

function main()
  a = Matrix(I, (1028, 1028))

  @time for i = 1:100
    a = Matrix(I, (1028, 1028))
    cho(a)
  end

  @time for i = 1:100
    a = Matrix(I, (1028, 1028))
    cho_inplace(a)
  end
end

main()


















And the same for solve (which is just the \ operator in Julia. You need to start the line of code with @. to say that it is in place


using LinearAlgebra
function solve(a, b)
  return a \ b
end

function solve_inplace(a,b )
   @. a \ b
end

function main()
  a = [1 0; 1 -2]
  b = [32; -4]


  @time for i = 1:100
    solve(a, b)
  end

  @time for i = 1:100
    solve_inplace(a, b)
  end
end

main()




CHOLESKY:

using LinearAlgebra


function cho(a)
  return cholesky(a)
end

function cho_inplace(a)
  return cholesky!(a)
end

function main()
  a = Matrix(I, (1028, 1028))

  @time for i = 1:100
    a = Matrix(I, (1028, 1028))
    cho(a)
  end

  @time for i = 1:100
    a = Matrix(I, (1028, 1028))
    cho_inplace(a)
  end
end

main()