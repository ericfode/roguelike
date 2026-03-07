import Lake
open Lake DSL

package «roguelike-formal» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib Roguelike where
  srcDir := "Roguelike"
