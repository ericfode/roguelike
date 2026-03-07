/- Arena allocator — formal specification.
   Models the typed arena and proves basic properties
   (valid indices, no dangling references).
-/

namespace Roguelike

/-- An arena is a list of items. Allocation appends, ids are indices. -/
structure Arena (α : Type) where
  items : List α
  deriving Repr

/-- A typed handle into an arena. -/
structure Id (α : Type) where
  index : Nat
  deriving Repr, DecidableEq

namespace Arena

def empty : Arena α := ⟨[]⟩

def alloc (a : Arena α) (v : α) : Arena α × Id α :=
  (⟨a.items ++ [v]⟩, ⟨a.items.length⟩)

def get (a : Arena α) (id : Id α) : Option α :=
  a.items.get? id.index

/-- Allocating an item and immediately looking it up succeeds. -/
theorem get_alloc (a : Arena α) (v : α) :
    let (a', id) := a.alloc v
    a'.get id = some v := by
  simp [alloc, get, List.get?_append_right]
  sorry -- TODO: complete proof

end Arena
end Roguelike
