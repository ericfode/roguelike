//! Typed arena allocator for game entities.
//! All game state lives in arenas — no scattered heap allocations.

use std::cell::Cell;

/// A typed arena that owns a contiguous block of `T` values.
/// Entities are allocated into the arena and referenced by `Id<T>`.
pub struct Arena<T> {
    items: Vec<T>,
}

/// A typed handle into an arena. Copy, lightweight, generation-free for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id<T> {
    index: u32,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn alloc(&mut self, value: T) -> Id<T> {
        let index = self.items.len() as u32;
        self.items.push(value);
        Id {
            index,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn get(&self, id: Id<T>) -> Option<&T> {
        self.items.get(id.index as usize)
    }

    pub fn get_mut(&mut self, id: Id<T>) -> Option<&mut T> {
        self.items.get_mut(id.index as usize)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}
