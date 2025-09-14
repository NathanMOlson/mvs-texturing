#ifndef __SPARSE_TABLE_H__
#define __SPARSE_TABLE_H__

#include <vector>

/**
  * Class representing a sparse table optimized for row and column wise access.
  */
template <typename C, typename R, typename T>
class SparseTable {
    public:
        typedef std::vector<std::pair<R, T> > Column;
        typedef std::vector<std::pair<C, T> > Row;

    private:
        std::vector<Column> column_wise_data;
        std::vector<Row> row_wise_data;

        std::size_t nnz;
    public:
        SparseTable();
        SparseTable(C cols, R rows);

        C cols() const;
        R rows() const;

        Column const & col(C id) const;
        Row const & row(R id) const;

        void set_value(C col, R row, T value);

        std::size_t get_nnz(void) const;
};

template <typename C, typename R, typename T> std::size_t
SparseTable<C, R, T>::get_nnz(void) const {
    return nnz;
}

template <typename C, typename R, typename T> C
SparseTable<C, R, T>::cols() const {
    return column_wise_data.size();
}

template <typename C, typename R, typename T> R
SparseTable<C, R, T>::rows() const {
    return row_wise_data.size();
}

template <typename C, typename R, typename T> typename SparseTable<C, R, T>::Column const &
SparseTable<C, R, T>::col(C id) const {
    return column_wise_data[id];
}

template <typename C, typename R, typename T> typename SparseTable<C, R, T>::Row const &
SparseTable<C, R, T>::row(R id) const {
    return row_wise_data[id];
}

template <typename C, typename R, typename T>
SparseTable<C, R, T>::SparseTable() {
    nnz = 0;
}

template <typename C, typename R, typename T>
SparseTable<C, R, T>::SparseTable(C cols, R rows) {
    column_wise_data.resize(cols);
    row_wise_data.resize(rows);
    nnz = 0;
}

template <typename C, typename R, typename T> void
SparseTable<C, R, T>::set_value(C col, R row, T value) {
    column_wise_data[col].push_back(std::pair<R, T>(row, value));
    row_wise_data[row].push_back(std::pair<C, T>(col, value));
    nnz++;
}

#endif 
