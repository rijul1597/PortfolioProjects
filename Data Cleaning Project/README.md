# 🧹 Nashville Housing – Data-Cleaning in T-SQL

This folder holds one SQL script that converts the raw **`NashvilleHousing`** table
into an analytics-ready dataset.

| Step | Transformation | Key SQL |
|------|----------------|---------|
| 1 | **Standardise dates** – cast `SaleDate` from `nvarchar` to native `DATE`. | `UPDATE … SET SaleDate = CONVERT(Date, SaleDate)` |
| 2 | **Fallback column** – add `SaleDateConverted` if the in-place cast fails. | `ALTER TABLE … ADD SaleDateConverted DATE` |
| 3 | **Fill missing `PropertyAddress`** from another row with the same `ParcelID`. | self-`JOIN` on `ParcelID` |
| 4 | **Split `PropertyAddress`** into `PropertySplitAddress` & `PropertySplitCity`. | `SUBSTRING` / `CHARINDEX` |
| 5 | **Split `OwnerAddress`** into `OwnerSplitAddress`, `OwnerSplitCity`, `OwnerSplitState`. | `PARSENAME(REPLACE(OwnerAddress, ',', '.') , n)` |
| 6 | **Normalise `SoldAsVacant`** – convert `'Y'/'N'` to `'Yes'/'No'`. | `CASE WHEN …` |
| 7 | **Remove duplicates** with a CTE + `ROW_NUMBER()`. | `WITH RowNumCTE AS (…) DELETE … WHERE row_num > 1` |
| 8 | **Drop unused columns** once data is clean. | `ALTER TABLE … DROP COLUMN …` |

Each block of SQL is idempotent, so you can rerun the script safely.

---

## ▶️ How to run

1. **Open SQL Server Management Studio (SSMS)** or Azure Data Studio.  
2. Connect to the database that contains `NashvilleHousing`.  
3. Execute:

```sql
USE PortfolioProject;      -- change if your DB name differs
:r clean_nashville.sql     -- SSMS command to run the script in this folder
```  
## 📈 Result snapshot (Excel upload Jun 24 2025)

| | Raw table | Cleaned table |
|---|-----------|--------------|
| Row count | 56 477 | 56 374 |
| % NULL `PropertyAddress` | 0.05 % | 0 % |
| Duplicate rows | 103 | 0 |
| Distinct `SaleDate` datatypes | 1 (`DATE`) | 1 (`DATE`) |


📜 License

MIT – see repository root LICENSE.
