using PhasorNetworks
using Documenter

DocMeta.setdocmeta!(PhasorNetworks, :DocTestSetup, :(using PhasorNetworks); recursive=true)

makedocs(;
    modules=[PhasorNetworks],
    authors="Wilkie Olin-Ammentorp",
    sitename="PhasorNetworks.jl",
    format=Documenter.HTML(;
        canonical="https://wilkieolin.github.io/PhasorNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "API Reference" => [
            "Types" => "api/types.md",
            "Networks" => "api/network.md",
            "Activations" => "api/activations.md",
            "Spiking" => "api/spiking.md",
            "Domains" => "api/domains.md",
            "VSA" => "api/vsa.md",
            "GPU" => "api/gpu.md",
            "Metrics" => "api/metrics.md",
            "SSM" => "api/ssm.md",
            "Attractor SSM" => "api/attractor_ssm.md",
            "Datasets" => "api/datasets.md",
            "Backend" => "api/backend.md",
            "Equilibrium Propagation" => "api/ep.md",
            "Holomorphic EP" => "api/hep.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/wilkieolin/PhasorNetworks.jl",
    devbranch="main",
)
