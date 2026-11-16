using Biwhitening
using Documenter

DocMeta.setdocmeta!(Biwhitening, :DocTestSetup, :(using Biwhitening); recursive=true)

makedocs(;
    modules=[Biwhitening],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="Biwhitening.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/Biwhitening.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/Biwhitening.jl",
    devbranch="main",
)
