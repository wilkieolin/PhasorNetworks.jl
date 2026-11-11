# scripts/e1_free_vs_nudge.jl — E1 follow-up 2: which settle sets the constraint?
#
# e1_settle_convergence.jl showed that the catastrophic EP failures on trained
# weights are a settle-length effect: identical parameters and batch give
# cos = -0.05 at T = 400 and +1.000 at T = 800. It also showed the FREE settle
# is stationary to 1e-9 in a cell whose gradient is inverted, so free-settle
# convergence is not the binding constraint and `settle_residual` cannot be
# used as the diagnostic.
#
# The remaining candidate is the NUDGED settle. StaticEP runs T_nudge steps from
# the free equilibrium with beta != 0; if that has not itself equilibrated, the
# Hebbian difference is taken between a converged state and an unconverged one,
# which is a bias no amount of beta-shrinking removes.
#
# This grids T_free against T_nudge independently at the parameter points that
# failed. If failures track T_nudge at fixed T_free, the nudged settle is the
# constraint and the guidance is "T_nudge must be scaled with T_free, not fixed
# at half of it". If they track T_free, it is the free settle after all and the
# stationarity probe needs replacing rather than supplementing.
#
# Oracle is fixed at T = 3200 for every cell so the reference is common.
#
# Run: julia --project=. -t auto scripts/e1_free_vs_nudge.jl

using PhasorNetworks, Lux, LinearAlgebra, Random, Statistics, Printf, Serialization
using Random: Xoshiro

const DIR = joinpath(@__DIR__, "..", "results", "ep_trained_vs_rescaled")
_envi(k, d) = parse(Int, get(ENV, k, string(d)))
_envf(k, d) = parse(Float32, get(ENV, k, string(d)))

const HID=_envi("E1_HID",256); const DOUT=_envi("E1_DOUT",64)
const SEED=_envi("E1_SEED",7); const DT=_envf("E1_DT",0.5)
const K_FD=_envi("E1N_K_FD",48); const DIR_SEED=_envi("E1_DIR_SEED",90210)
const PROBE_B=_envi("E1_PROBE_B",16)
const FD_EPS=_envf("E1_FD_EPS",0.03)
const BETA=_envf("E1N_BETA",0.1)
const ORACLE_T=_envi("E1N_ORACLE_T",3200)
const T_FREES  = [200, 400, 800, 1600]
const T_NUDGES = [100, 200, 400, 800]

# The three cells that failed, as (run, epoch, batch).
const CELLS = [("nodecay", 8, 1), ("nodecay", 14, 1), ("wd1e-4", 4, 3)]

function encode_phase(imgs)
    flat = reshape(imgs, :, size(imgs,3)); μ=mean(flat;dims=1); σ=std(flat;dims=1).+1f-6
    return Phase.(0.5f0 .* tanh.((flat .- μ) ./ σ))
end
make_codes(rng)=ComplexF32.(angle_to_complex(orthogonal_codes(rng,DOUT,10)))
direction(k,n)=(d=randn(Xoshiro(DIR_SEED+k),Float32,n); d./=norm(d); d)
cos_sim(a,b)=dot(a,b)/(norm(a)*norm(b)+1e-12)

function build_chain(rng)
    ch=Chain(PhasorDense(784=>HID,normalize_to_unit_circle,use_bias=true),
             PhasorDense(HID=>DOUT,normalize_to_unit_circle,use_bias=true))
    ps,st=Lux.setup(rng,ch); return ch,ps,st
end

function fd_directional(chain,ps,st,x,cost,K;eps,T)
    W=ps.layer_1.weight; n=length(W); W0=copy(W); out=zeros(Float32,K)
    loss_at()=ep_loss(cost,phasor_settle(chain,ps,st,x,cost,0f0;T=T,dt=DT)[end])
    for k in 1:K
        d=reshape(direction(k,n),size(W))
        @. W=W0+eps*d; Lp=loss_at(); @. W=W0-eps*d; Lm=loss_at()
        out[k]=(Lp-Lm)/(2eps)
    end
    W.=W0; return out
end

function main()
    tr=fashion_mnist_data(:train)
    X=encode_phase(Float32.(tr.features[:,:,1:48])); y=Int.(tr.targets[1:48]).+1
    codes=make_codes(Xoshiro(SEED+1)); chain,_,st=build_chain(Xoshiro(SEED))
    snaps=Dict(r=>deserialize(joinpath(DIR,"snapshots_$r.jls")) for r in ("nodecay","wd1e-4"))

    rows=NamedTuple[]
    for (run,ep,b) in CELLS
        entry=first(s for s in snaps[run] if s[1]==ep); ps=entry[2]
        n=length(ps.layer_1.weight); n1=norm(ps.layer_1.weight)
        sl=((b-1)*PROBE_B+1):(b*PROBE_B); x=X[:,sl]; cost=CodebookCost(codes,y[sl])
        ref=fd_directional(chain,ps,st,x,cost,K_FD;eps=FD_EPS,T=ORACLE_T)
        @printf("\n=== %s epoch %d batch %d | ‖W₁‖=%.1f | oracle T=%d ===\n",
                run,ep,b,n1,ORACLE_T)
        @printf("%9s |%s\n","T_free",join((@sprintf("%9d",tn) for tn in T_NUDGES),""))
        @printf("%9s |%s   (cos, one-sided β=%.2f)\n","T_nudge→","-"^(9*length(T_NUDGES)),BETA)
        for tf in T_FREES
            line=@sprintf("%9d |",tf)
            for tn in T_NUDGES
                m=StaticEP(β=BETA,T_free=tf,T_nudge=tn,dt=DT,centered=false)
                g,_=ep_gradient(m,chain,ps,st,x,cost); gv=vec(g.layer_1.weight)
                c=cos_sim(Float32[dot(gv,direction(k,n)) for k in 1:K_FD],ref)
                line*=@sprintf("%9.3f",c)
                push!(rows,(;run,epoch=ep,batch=b,w1_norm=n1,T_free=tf,T_nudge=tn,
                             cos_fd=c,beta=BETA,k_fd=K_FD,oracle_T=ORACLE_T))
            end
            println(line); flush(stdout)
        end
    end
    path=joinpath(DIR,"free_vs_nudge.csv")
    open(path,"w") do io
        println(io,join(string.(keys(rows[1])),","))
        for r in rows
            println(io,join([v isa AbstractFloat ? @sprintf("%.6g",v) : string(v) for v in values(r)],","))
        end
    end
    @info "wrote $path ($(length(rows)) rows)"
end
main()
