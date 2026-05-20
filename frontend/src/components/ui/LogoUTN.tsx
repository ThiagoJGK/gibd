import React from 'react';

interface LogoUTNProps extends React.SVGProps<SVGSVGElement> {}

export function LogoUTN(props: LogoUTNProps) {
  const { style, ...restProps } = props;
  
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 135.46667 135.46667"
      style={{ overflow: 'visible', ...style }}
      {...restProps}
    >
      <g id="layer1" transform="translate(0,-235)">
        <g
          id="g2294"
          transform="matrix(1.8825461,0,0,1.8825461,-110.89689,-54.949991)"
        >
          {/* Central Vertical Column */}
          <rect
            y={159.92024}
            x={90.405602}
            height={59.525364}
            width={9.5253668}
            id="rect815"
            opacity={1}
            fill="currentColor"
            fillOpacity={1}
            stroke="currentColor"
            strokeWidth={0.474633}
            strokeLinejoin="round"
            strokeMiterlimit={4}
            strokeDasharray="none"
            strokeDashoffset={0}
            strokeOpacity={1}
            paintOrder="normal"
          />
          {/* Central Horizontal Bar */}
          <rect
            y={184.62465}
            x={70.099335}
            height={9.3164644}
            width={49.316463}
            id="rect817"
            opacity={1}
            fill="currentColor"
            fillOpacity={1}
            stroke="currentColor"
            strokeWidth={0.683535}
            strokeLinejoin="round"
            strokeMiterlimit={4}
            strokeDasharray="none"
            strokeDashoffset={0}
            strokeOpacity={1}
            paintOrder="normal"
          />
          {/* Top Arch Shape */}
          <path
            id="path819"
            d="m 118.7286,160.4028 -7.58816,0.0475 A 16.000629,17 0 0 1 95.277845,176.65091 16.000629,17 0 0 1 79.211641,160.65085 l -8.444963,0.0532 a 23.981857,23.980871 0 0 0 24.130312,23.67917 23.981857,23.980871 0 0 0 23.83161,-23.98045 z"
            opacity={1}
            fill="currentColor"
            fillOpacity={1}
            stroke="currentColor"
            strokeWidth={2.03826}
            strokeLinejoin="round"
            strokeMiterlimit={4}
            strokeDasharray="none"
            strokeDashoffset={0}
            strokeOpacity={1}
            paintOrder="normal"
          />
          {/* Bottom Arch Shape */}
          <path
            id="path819-7"
            d="m 118.36166,218.60107 -7.58816,-0.0475 A 16.000629,17 0 0 0 94.9109,202.35296 16.000629,17 0 0 0 78.8447,218.35302 l -8.44497,-0.0532 a 23.981857,23.980871 0 0 1 24.13032,-23.67917 23.981857,23.980871 0 0 1 23.83161,23.98045 z"
            opacity={1}
            fill="currentColor"
            fillOpacity={1}
            stroke="currentColor"
            strokeWidth={2.03826}
            strokeLinejoin="round"
            strokeMiterlimit={4}
            strokeDasharray="none"
            strokeDashoffset={0}
            strokeOpacity={1}
            paintOrder="normal"
          />
        </g>
      </g>
    </svg>
  );
}
