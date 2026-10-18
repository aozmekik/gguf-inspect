//! Lookup by name without storing every name twice.
//!
//! A `HashMap<String, usize>` next to the list of entries would copy every
//! key and tensor name and cost about 40 bytes of table per entry. A sorted
//! list of positions costs 8 bytes per entry and answers lookups by binary
//! search, which is fast enough for the few dozen metadata keys and few
//! thousand tensors of real files.

/// Positions into a list of named items, sorted by name.
#[derive(Debug, Clone, Default)]
pub(crate) struct NameIndex(Vec<usize>);

impl NameIndex {
    /// Sorts the positions `0..len` by `name(position)`.
    ///
    /// Returns `Err` with the position of the first item, in list order,
    /// whose name repeats an earlier one.
    pub(crate) fn build<'a>(len: usize, name: impl Fn(usize) -> &'a str) -> Result<Self, usize> {
        let mut positions: Vec<usize> = (0..len).collect();
        // Ties broken by position, so equal names sit in list order.
        positions.sort_unstable_by(|&a, &b| name(a).cmp(name(b)).then(a.cmp(&b)));
        let first_repeat = positions
            .windows(2)
            .filter(|pair| name(pair[0]) == name(pair[1]))
            .map(|pair| pair[1])
            .min();
        match first_repeat {
            Some(position) => Err(position),
            None => Ok(Self(positions)),
        }
    }

    /// The position of the item called `wanted`.
    pub(crate) fn find<'a>(&self, wanted: &str, name: impl Fn(usize) -> &'a str) -> Option<usize> {
        self.0
            .binary_search_by(|&position| name(position).cmp(wanted))
            .ok()
            .map(|found| self.0[found])
    }

    /// Adds the item at `position`, unless its name is already present.
    #[cfg(test)]
    pub(crate) fn insert<'a>(&mut self, position: usize, name: impl Fn(usize) -> &'a str) -> bool {
        let key = name(position);
        match self.0.binary_search_by(|&p| name(p).cmp(key)) {
            Ok(_) => false,
            Err(at) => {
                self.0.insert(at, position);
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_names_and_reports_the_first_repeat_in_list_order() {
        let names = ["b", "a", "c"];
        let index = NameIndex::build(names.len(), |i| names[i]).unwrap();
        assert_eq!(index.find("a", |i| names[i]), Some(1));
        assert_eq!(index.find("c", |i| names[i]), Some(2));
        assert_eq!(index.find("d", |i| names[i]), None);

        // "x" repeats at 3, "a" at 4: the first repeat in list order is 3.
        let names = ["a", "x", "b", "x", "a", "x"];
        assert_eq!(NameIndex::build(names.len(), |i| names[i]).unwrap_err(), 3);
        assert!(NameIndex::build(0, |_| "").is_ok());
    }

    #[test]
    fn insert_keeps_the_index_sorted() {
        let names = ["m", "a", "z", "a"];
        let mut index = NameIndex::default();
        for position in 0..3 {
            assert!(index.insert(position, |i| names[i]));
        }
        assert!(!index.insert(3, |i| names[i]));
        assert_eq!(index.find("z", |i| names[i]), Some(2));
    }
}
