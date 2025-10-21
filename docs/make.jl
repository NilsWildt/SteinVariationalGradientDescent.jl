using Documenter
using SteinVariationalGradientDescent

DocMeta.setdocmeta!(SteinVariationalGradientDescent, :DocTestSetup,
                    :(using SteinVariationalGradientDescent); recursive=true)

makedocs(
    sitename = "SteinVariationalGradientDescent.jl",
    modules = [SteinVariationalGradientDescent],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
    ],
)
