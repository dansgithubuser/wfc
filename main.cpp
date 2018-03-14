#if 0
	# Who needs a makefile? Just run ./main.cpp [arguments]
	# ./main.cpp clean will clear the build dir.
	set -eu

	if [ ! -z ${1+x} ] && [ $1 == "clean" ]; then
		rm -rf build
		exit 0
	fi

	git submodule update --init --recursive

	mkdir -p build

	CXX=g++
	CPPFLAGS="--std=c++14 -Wall -Wno-sign-compare -O2 -g -DNDEBUG"
	LDLIBS="-lstdc++ -lpthread -ldl"
	OBJECTS=""

	for source_path in *.cpp; do
		obj_path="build/${source_path%.cpp}.o"
		OBJECTS="$OBJECTS $obj_path"
		if [ ! -f $obj_path ] || [ $obj_path -ot $source_path ]; then
			echo "Compiling $source_path to $obj_path..."
			$CXX $CPPFLAGS                      \
			    -I libs -I libs/emilib          \
			    -c $source_path -o $obj_path
		fi
	done

	echo "Linking..."
	$CXX $CPPFLAGS $OBJECTS $LDLIBS -o wfc.bin

	# Run it:
	mkdir -p output
	./wfc.bin $@
	exit
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <emilib/strprintf.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

#define CHECK(CHECK_EXPRESSION)\
	do\
	{\
		if(!(CHECK_EXPRESSION))\
		{\
			throw std::logic_error("CHECK FAILED: " #CHECK_EXPRESSION);\
		}\
	} while (false)

#include "arrays.hpp"

#define FORI(FORI_ITERATOR, FORI_MAX)\
	auto FORI_MAX_VALUE_##FORI_ITERATOR=FORI_MAX;\
	for(auto FORI_ITERATOR=decltype(FORI_MAX)(0); FORI_ITERATOR<FORI_MAX_VALUE_##FORI_ITERATOR; ++FORI_ITERATOR)

struct RGBA
{
	uint8_t r, g, b, a;
};
static_assert(sizeof(RGBA) == 4, "");
bool operator==(RGBA x, RGBA y) { return x.r == y.r && x.g == y.g && x.b == y.b && x.a == y.a; }

using Bool              = uint8_t; // To avoid problems with vector<bool>
using ColorIndex        = uint8_t; // tile index or color index. If you have more than 255, don't.
using Palette           = std::vector<RGBA>;
using Pattern           = std::vector<ColorIndex>;
using PatternHash       = uint64_t; // Another representation of a Pattern.
using PatternPrevalence = std::unordered_map<PatternHash, size_t>;
using RandomDouble      = std::function<double()>;
using PatternIndex      = uint16_t;

const auto kInvalidIndex = static_cast<size_t>(-1);
const auto kInvalidHash = static_cast<PatternHash>(-1);

const size_t kUpscale             =   4; // Upscale images before saving

enum class Result
{
	kSuccess,
	kFail,
	kUnfinished,
};

const char* result2str(const Result result)
{
	return result == Result::kSuccess ? "success"
	     : result == Result::kFail    ? "fail"
	     : "unfinished";
}

const size_t MAX_COLORS = 1 << (sizeof(ColorIndex) * 8);

using Graphics = Array2D<std::vector<ColorIndex>>;

struct PalettedImage
{
	size_t                  width, height;
	std::vector<ColorIndex> data; // width * height
	Palette                 palette;

	ColorIndex at_wrapped(size_t x, size_t y) const
	{
		return data[width * (y % height) + (x % width)];
	}
};

// What actually changes
struct Output
{
	// _width X _height X num_patterns
	// _wave.get(x, y, t) == is the pattern t possible at x, y?
	// Starts off true everywhere.
	Array3D<Bool> _wave;
	Array2D<Bool> _changes; // _width X _height. Starts off false everywhere.
};

using Image = Array2D<RGBA>;

// ----------------------------------------------------------------------------

Image upsample(const Image& image)
{
	Image result(image.width() * kUpscale, image.height() * kUpscale, {});
	FORI (y, result.height()) {
		FORI (x, result.width()) {
			result.set(x, y, image.get(x / kUpscale, y / kUpscale));
		}
	}
	return result;
}

// ----------------------------------------------------------------------------

class Model
{
public:
	size_t              _width;      // Of output image.
	size_t              _height;     // Of output image.
	size_t              _num_patterns;
	bool                _periodic_out;
	size_t              _foundation = kInvalidIndex; // Index of pattern which is at the base, or kInvalidIndex

	// The weight of each pattern (e.g. how often that pattern occurs in the sample image).
	std::vector<double> _pattern_weight; // num_patterns

	virtual bool propagate(Output* output) const = 0;
	virtual bool on_boundary(int x, int y) const = 0;
	virtual Image image(const Output& output) const = 0;
};

// ----------------------------------------------------------------------------

class OverlappingModel : public Model
{
public:
	OverlappingModel(
		const PatternPrevalence& hashed_patterns,
		const Palette&           palette,
		int                      n,
		bool                     periodic_out,
		size_t                   width,
		size_t                   height,
		PatternHash              foundation_pattern);

	bool propagate(Output* output) const override;

	bool on_boundary(int x, int y) const override
	{
		return !_periodic_out && (x + _n > _width || y + _n > _height);
	}

	Image image(const Output& output) const override;

	Graphics graphics(const Output& output) const;

private:
	int                       _n;
	// num_patterns X (2 * n - 1) X (2 * n - 1) X ???
	// list of other pattern indices that agree on this x/y offset (?)
	Array3D<std::vector<PatternIndex>> _propagator;
	std::vector<Pattern>               _patterns;
	Palette                            _palette;
};

// ----------------------------------------------------------------------------

double calc_sum(const std::vector<double>& a)
{
	return std::accumulate(a.begin(), a.end(), 0.0);
}

// Pick a random index weighted by a
size_t spin_the_bottle(const std::vector<double>& a, double between_zero_and_one)
{
	double sum = calc_sum(a);

	if (sum == 0.0) {
		return std::floor(between_zero_and_one * a.size());
	}

	double between_zero_and_sum = between_zero_and_one * sum;

	double accumulated = 0;

	FORI (i, a.size()) {
		accumulated += a[i];
		if (between_zero_and_sum <= accumulated) {
			return i;
		}
	}

	return 0;
}

PatternHash hash_from_pattern(const Pattern& pattern, size_t palette_size)
{
	CHECK(std::pow((double)palette_size, (double)pattern.size()) < std::pow(2.0, sizeof(PatternHash) * 8));
	PatternHash result = 0;
	size_t power = 1;
	FORI (i, pattern.size())
	{
		result += pattern[pattern.size() - 1 - i] * power;
		power *= palette_size;
	}
	return result;
}

Pattern pattern_from_hash(const PatternHash hash, int n, size_t palette_size)
{
	size_t residue = hash;
	size_t power = std::pow(palette_size, n * n);
	Pattern result(n * n);

	for (size_t i = 0; i < result.size(); ++i)
	{
		power /= palette_size;
		size_t count = 0;

		while (residue >= power)
		{
			residue -= power;
			count++;
		}

		result[i] = static_cast<ColorIndex>(count);
	}

	return result;
}

template<typename Fun>
Pattern make_pattern(int n, const Fun& fun)
{
	Pattern result(n * n);
	FORI (dy, n) {
		FORI (dx, n) {
			result[dy * n + dx] = fun(dx, dy);
		}
	}
	return result;
};

// ----------------------------------------------------------------------------

OverlappingModel::OverlappingModel(
	const PatternPrevalence& hashed_patterns,
	const Palette&           palette,
	int                      n,
	bool                     periodic_out,
	size_t                   width,
	size_t                   height,
	PatternHash              foundation_pattern)
{
	_width        = width;
	_height       = height;
	_num_patterns = hashed_patterns.size();
	_periodic_out = periodic_out;
	_n            = n;
	_palette      = palette;

	for (const auto& it : hashed_patterns) {
		if (it.first == foundation_pattern) {
			_foundation = _patterns.size();
		}

		_patterns.push_back(pattern_from_hash(it.first, n, _palette.size()));
		_pattern_weight.push_back(it.second);
	}

	const auto agrees = [&](const Pattern& p1, const Pattern& p2, int dx, int dy) {
		int xmin = dx < 0 ? 0 : dx, xmax = dx < 0 ? dx + n : n;
		int ymin = dy < 0 ? 0 : dy, ymax = dy < 0 ? dy + n : n;
		for (int y = ymin; y < ymax; ++y) {
			for (int x = xmin; x < xmax; ++x) {
				if (p1[x + n * y] != p2[x - dx + n * (y - dy)]) {
					return false;
				}
			}
		}
		return true;
	};

	_propagator = Array3D<std::vector<PatternIndex>>(_num_patterns, 2 * n - 1, 2 * n - 1, {});

	size_t longest_propagator = 0;
	size_t sum_propagator = 0;

	FORI (t, _num_patterns) {
		FORI (x, 2 * n - 1) {
			FORI (y, 2 * n - 1) {
				auto& list = _propagator.mut_ref(t, x, y);
				FORI (t2, _num_patterns) {
					if (agrees(_patterns[t], _patterns[t2], x - n + 1, y - n + 1)) {
						list.push_back(t2);
					}
				}
				list.shrink_to_fit();
				longest_propagator = std::max(longest_propagator, list.size());
				sum_propagator += list.size();
			}
		}
	}
}

bool OverlappingModel::propagate(Output* output) const
{
	bool did_change = false;

	for (int x1 = 0; x1 < _width; ++x1) {
		for (int y1 = 0; y1 < _height; ++y1) {
			if (!output->_changes.get(x1, y1)) { continue; }
			output->_changes.set(x1, y1, false);

			for (int dx = -_n + 1; dx < _n; ++dx) {
				for (int dy = -_n + 1; dy < _n; ++dy) {
					auto x2 = x1 + dx;
					auto y2 = y1 + dy;

					auto sx = x2;
					if      (sx <  0)      { sx += _width; }
					else if (sx >= _width) { sx -= _width; }

					auto sy = y2;
					if      (sy <  0)       { sy += _height; }
					else if (sy >= _height) { sy -= _height; }

					if (!_periodic_out && (sx + _n > _width || sy + _n > _height)) {
						continue;
					}

					for (int t2 = 0; t2 < _num_patterns; ++t2) {
						if (!output->_wave.get(sx, sy, t2)) { continue; }

						bool can_pattern_fit = false;

						const auto& prop = _propagator.ref(t2, _n - 1 - dx, _n - 1 - dy);
						for (const auto& t3 : prop) {
							if (output->_wave.get(x1, y1, t3)) {
								can_pattern_fit = true;
								break;
							}
						}

						if (!can_pattern_fit) {
							output->_changes.set(sx, sy, true);
							output->_wave.set(sx, sy, t2, false);
							did_change = true;
						}
					}
				}
			}
		}
	}

	return did_change;
}

Graphics OverlappingModel::graphics(const Output& output) const
{
	Graphics result(_width, _height, {});
	FORI (y, _height) {
		FORI (x, _width) {
			auto& tile_constributors = result.mut_ref(x, y);

			for (int dy = 0; dy < _n; ++dy) {
				for (int dx = 0; dx < _n; ++dx) {
					int sx = x - dx;
					if (sx < 0) sx += _width;

					int sy = y - dy;
					if (sy < 0) sy += _height;

					if (on_boundary(sx, sy)) { continue; }

					for (int t = 0; t < _num_patterns; ++t) {
						if (output._wave.get(sx, sy, t)) {
							tile_constributors.push_back(_patterns[t][dx + dy * _n]);
						}
					}
				}
			}
		}
	}
	return result;
}

Image image_from_graphics(const Graphics& graphics, const Palette& palette)
{
	Image result(graphics.width(), graphics.height(), {0, 0, 0, 0});

	FORI (y, graphics.height()) {
		FORI (x, graphics.width()) {
			const auto& tile_constributors = graphics.ref(x, y);
			if (tile_constributors.empty()) {
				result.set(x, y, {0, 0, 0, 255});
			} else if (tile_constributors.size() == 1) {
				result.set(x, y, palette[tile_constributors[0]]);
			} else {
				size_t r = 0;
				size_t g = 0;
				size_t b = 0;
				size_t a = 0;
				for (const auto tile : tile_constributors) {
					r += palette[tile].r;
					g += palette[tile].g;
					b += palette[tile].b;
					a += palette[tile].a;
				}
				r /= tile_constributors.size();
				g /= tile_constributors.size();
				b /= tile_constributors.size();
				a /= tile_constributors.size();
				result.set(x, y, {(uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a});
			}
		}
	}

	return result;
}

Image OverlappingModel::image(const Output& output) const
{
	return upsample(image_from_graphics(graphics(output), _palette));
}

// ----------------------------------------------------------------------------

PalettedImage load_paletted_image(const std::string& path)
{
	int width, height, comp;
	RGBA* rgba = reinterpret_cast<RGBA*>(stbi_load(path.c_str(), &width, &height, &comp, 4));
	CHECK(rgba);
	const auto num_pixels = width * height;

	// Fix issues with stbi_load:
	if (comp == 1) {
		// input was greyscale - set alpha:
		for (auto pixel = rgba; pixel < rgba + num_pixels; ++pixel) {
			pixel->a = pixel->r;
		}
	} else {
		if (comp == 3) {
			for (auto pixel = rgba; pixel < rgba + num_pixels; ++pixel) {
				pixel->a = 255;
			}
		}
		for (auto pixel = rgba; pixel< rgba + num_pixels; ++pixel) {
			if (pixel->a == 0) {
				*pixel = RGBA{0,0,0,0};
			}
		}
	}

	std::vector<RGBA> palette;
	std::vector<ColorIndex> data;

	FORI (pixel_idx, num_pixels) {
		const RGBA color = rgba[pixel_idx];
		const auto color_idx = std::find(palette.begin(), palette.end(), color) - palette.begin();
		if (color_idx == palette.size()) {
			CHECK(palette.size() < MAX_COLORS);
			palette.push_back(color);
		}
		data.push_back(color_idx);
	}

	stbi_image_free(rgba);

	return PalettedImage{
		static_cast<size_t>(width),
		static_cast<size_t>(height),
		data, palette
	};
}

// n = side of the pattern, e.g. 3.
PatternPrevalence extract_patterns(
	const PalettedImage& sample, int n, bool periodic_in, size_t symmetry,
	PatternHash* out_lowest_pattern)
{
	CHECK(n < sample.width);
	CHECK(n < sample.height);

	const auto pattern_from_sample = [&](size_t x, size_t y) {
		return make_pattern(n, [&](size_t dx, size_t dy){ return sample.at_wrapped(x + dx, y + dy); });
	};
	const auto rotate  = [&](const Pattern& p){ return make_pattern(n, [&](size_t x, size_t y){ return p[n - 1 - y + x * n]; }); };
	const auto reflect = [&](const Pattern& p){ return make_pattern(n, [&](size_t x, size_t y){ return p[n - 1 - x + y * n]; }); };

	PatternPrevalence patterns;

	FORI (y, periodic_in ? sample.height : sample.height - n + 1) {
		FORI (x, periodic_in ? sample.width : sample.width - n + 1) {
			std::array<Pattern, 8> ps;
			ps[0] = pattern_from_sample(x, y);
			ps[1] = reflect(ps[0]);
			ps[2] = rotate(ps[0]);
			ps[3] = reflect(ps[2]);
			ps[4] = rotate(ps[2]);
			ps[5] = reflect(ps[4]);
			ps[6] = rotate(ps[4]);
			ps[7] = reflect(ps[6]);

			for (int k = 0; k < symmetry; ++k) {
				auto hash = hash_from_pattern(ps[k], sample.palette.size());
				patterns[hash] += 1;
				if (out_lowest_pattern && y == sample.height - 1) {
					*out_lowest_pattern = hash;
				}
			}
		}
	}

	return patterns;
}

Result find_lowest_entropy(const Model& model, const Output& output, RandomDouble& random_double,
                           int* argminx, int* argminy)
{
	// We actually calculate exp(entropy), i.e. the sum of the weights of the possible patterns

	double min = std::numeric_limits<double>::infinity();

	for (int x = 0; x < model._width; ++x) {
		for (int y = 0; y < model._height; ++y) {
			if (model.on_boundary(x, y)) { continue; }

			size_t num_superimposed = 0;
			double entropy = 0;

			for (int t = 0; t < model._num_patterns; ++t) {
				if (output._wave.get(x, y, t)) {
					num_superimposed += 1;
					entropy += model._pattern_weight[t];
				}
			}

			if (entropy == 0 || num_superimposed == 0) {
				return Result::kFail;
			}

			if (num_superimposed == 1) {
				continue; // Already frozen
			}

			// Add a tie-breaking bias:
			const double noise = 0.5 * random_double();
			entropy += noise;

			if (entropy < min) {
				min = entropy;
				*argminx = x;
				*argminy = y;
			}
		}
	}

	if (min == std::numeric_limits<double>::infinity()) {
		return Result::kSuccess;
	} else {
		return Result::kUnfinished;
	}
}

Result observe(const Model& model, Output* output, RandomDouble& random_double)
{
	int argminx, argminy;
	const auto result = find_lowest_entropy(model, *output, random_double, &argminx, &argminy);
	if (result != Result::kUnfinished) { return result; }

	std::vector<double> distribution(model._num_patterns);
	for (int t = 0; t < model._num_patterns; ++t) {
		distribution[t] = output->_wave.get(argminx, argminy, t) ? model._pattern_weight[t] : 0;
	}
	size_t r = spin_the_bottle(std::move(distribution), random_double());
	for (int t = 0; t < model._num_patterns; ++t) {
		output->_wave.set(argminx, argminy, t, t == r);
	}
	output->_changes.set(argminx, argminy, true);

	return Result::kUnfinished;
}

Output create_output(const Model& model)
{
	Output output;
	output._wave = Array3D<Bool>(model._width, model._height, model._num_patterns, true);
	output._changes = Array2D<Bool>(model._width, model._height, false);

	if (model._foundation != kInvalidIndex) {
		FORI (x, model._width) {
			FORI (t, model._num_patterns) {
				if (t != model._foundation) {
					output._wave.set(x, model._height - 1, t, false);
				}
			}
			output._changes.set(x, model._height - 1, true);

			FORI (y, model._height - 1) {
				output._wave.set(x, y, model._foundation, false);
				output._changes.set(x, y, true);
			}

			while (model.propagate(&output));
		}
	}

	return output;
}

Image scroll_diagonally(const Image& image)
{
	const auto width = image.width();
	const auto height = image.height();
	Image result(width, height);
	FORI (y, height) {
		FORI (x, width) {
			result.set(x, y, image.get((x + 1) % width, (y + 1) % height));
		}
	}
	return result;
}

Result run(Output* output, const Model& model, size_t seed, size_t limit)
{
	std::mt19937 gen(seed);
	std::uniform_real_distribution<double> dis(0.0, 1.0);
	RandomDouble random_double = [&]() { return dis(gen); };

	for (size_t l = 0; l < limit || limit == 0; ++l) {
		Result result = observe(model, output, random_double);

		if (result != Result::kUnfinished) {
			return result;
		}
		while (model.propagate(output));
	}
	return Result::kUnfinished;
}

void run_and_write(const std::string& name, const Model& model)
{
	const size_t limit       = 0;
	const size_t screenshots = 2;

	FORI (i, screenshots) {
		FORI (attempt, 10) {
			(void)attempt;
			int seed = rand();

			Output output = create_output(model);

			const auto result = run(&output, model, seed, limit);

			if (result == Result::kSuccess) {
				const auto image = model.image(output);
				const auto out_path = emilib::strprintf("output/%s_%lu.png", name.c_str(), i);
				CHECK(stbi_write_png(out_path.c_str(), image.width(), image.height(), 4, image.data(), 0));
				break;
			}
		}
	}
}

std::unique_ptr<Model> make_overlapping()
{
	const int    n              = 3;
	const size_t out_width      = 48;
	const size_t out_height     = 48;
	const size_t symmetry       = 8;
	const bool   periodic_out   = true;
	const bool   periodic_in    = true;
	const auto   has_foundation = false;

	const auto sample_image = load_paletted_image("samples/simple_knot.bmp");
	PatternHash foundation = kInvalidHash;
	const auto hashed_patterns = extract_patterns(sample_image, n, periodic_in, symmetry, has_foundation ? &foundation : nullptr);

	return std::unique_ptr<Model>{
		new OverlappingModel{hashed_patterns, sample_image.palette, n, periodic_out, out_width, out_height, foundation}
	};
}

int main(int argc, char* argv[])
{
	const auto model = make_overlapping();
	run_and_write("simple_knot.bmp", *model);
}
