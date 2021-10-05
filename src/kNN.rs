use std::cell::RefCell;

#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub struct kNN<'a, 'b> {
    X: &'a ndarray::ArrayView2<'b, f64>,
    order: RefCell<Option<ndarray::Array2<f64>>>,
}

impl<'a, 'b> kNN<'a, 'b> {
    #[allow(dead_code)]
    pub fn new(X: &'a ndarray::ArrayView2<'b, f64>) -> kNN<'a, 'b> {
        kNN {
            X,
            order: RefCell::new(Option::None),
        }
    }

    #[allow(dead_code)]
    fn calculate_ordering(&self) -> ndarray::Array2<usize> {
        let n = self.X.nrows();
        let mut distances = ndarray::Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i >= j {
                    distances[[i, j]] = distances[[j, i]]
                } else {
                    for k in 0..self.X.ncols() {
                        distances[[i, j]] += (self.X[[i, k]] - self.X[[j, k]]).powi(2)
                    }
                }
            }
        }

        // A rather complex ordering = numpy.argsort(distances, 1)
        let mut ordering = ndarray::Array2::<usize>::default((n, n));
        for (i, mut row) in ordering.axis_iter_mut(ndarray::Axis(0)).enumerate() {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|a, b| {
                distances[[i, *a]].partial_cmp(&distances[[i, *b]]).unwrap()
            });
            for (j, val) in row.iter_mut().enumerate() {
                *val = order[j]
            }
        }

        ordering
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_X_ordering() {
        let X = ndarray::array![[1.], [1.5], [3.], [-0.5]];
        let X_view = X.view();

        let change_in_mean = kNN::new(&X_view);
        let ordering = change_in_mean.calculate_ordering();
        let expected = ndarray::array![[0, 1, 3, 2], [1, 0, 2, 3], [2, 1, 0, 3], [3, 0, 1, 2]];
        assert_eq!(ordering, expected)
    }
}
