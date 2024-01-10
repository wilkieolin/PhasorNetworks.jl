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
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/wilkieolin/PhasorNetworks.jl",
    devbranch="main",
)
