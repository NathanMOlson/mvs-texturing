#ifndef __SPARSE_TABLE_H__
#define __SPARSE_TABLE_H__

#include <vector>

/**
  * Class representing a sparse table optimized for column wise access.
  */
template <typename C, typename R, typename T>
class SparseTable {
    public:
        typedef std::vector<std::pair<R, T> > Column;

    private:
        std::vector<Column> column_wise_data;

    public:
        SparseTable();
        SparseTable(C cols);

        C cols() const;

        Column const & col(C id) const;

        void set_value(C col, R row, T value);
};

template <typename C, typename R, typename T> C
SparseTable<C, R, T>::cols() const {
    return column_wise_data.size();
}

template <typename C, typename R, typename T> typename SparseTable<C, R, T>::Column const &
SparseTable<C, R, T>::col(C id) const {
    return column_wise_data[id];
}

template <typename C, typename R, typename T>
SparseTable<C, R, T>::SparseTable() {
}

template <typename C, typename R, typename T>
SparseTable<C, R, T>::SparseTable(C cols) {
    column_wise_data.resize(cols);
}

template <typename C, typename R, typename T> void
SparseTable<C, R, T>::set_value(C col, R row, T value) {
    column_wise_data[col].push_back(std::pair<R, T>(row, value));
}

#endif 
