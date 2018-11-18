package filing

import (
	"os"
	"path/filepath"
)

//GetFilePaths returns a slice of strings of the path to all the files in the directory
func GetFilePaths(dir string) []string {
	paths := make([]string, 0)
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	return paths
}
